import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler

scheduler = DDPMScheduler(num_train_timesteps=1000)

def add_noise(x_0, t, noise=None):
    """
    给 x_0 在时间步 t 上添加噪声，返回 x_t, noise
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    x_t = scheduler.add_noise(x_0, noise, t.long())  # 这里调 scheduler
    return x_t, noise

class DiffusionTrainer:
    def __init__(self, model, dataset, val_dataset=None, lr=1e-4, batch_size=64, device="cuda:0"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None
        self.best_val_loss = float('inf')

    def train(self, num_epochs, val_dataset=None, save_path="best_model.pt"):
        if val_dataset is not None:
            self.val_dataloader = DataLoader(val_dataset, batch_size=self.dataloader.batch_size)
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                # batch 格式: (label, node_feature, pos_w, pos_d, r_id, adj_mx)
                # 6 个张量, each shape: [B, ...]
                label, node_feature, pos_w, pos_d, r_id, adj_mx = [b.to(self.device) for b in batch]

                # label = label.unsqueeze(1)  # [B, V, T] -> [B, 1, V, T]
                # node_feature = node_feature.unsqueeze(1)  # 同理

                # 构造模型需要的 condition: (node_feature, pos_w, pos_d)
                cond = (node_feature, pos_w, pos_d)

                # 采样随机扩散时间步 t
                B = label.size(0)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=self.device, dtype=torch.long)

                # 给 label 添加噪声 => (x_t, noise)
                # label 形如 [B, F, V, T_p], add_noise 同维度
                x_t, noise = add_noise(label, t)

                # 前向: pred_noise = model(x_t, t, cond)
                pred_noise = self.model(x_t, t, cond)
                assert pred_noise[:, :, :, -label.shape[-1]:].shape == noise.shape, f"Mismatch in shape: pred_noise slice {pred_noise[:, :, :, -label.shape[-1]:].shape} vs noise {noise.shape}"

                # 计算噪声预测的 MSE
                loss = F.mse_loss(pred_noise[:, :, :, -label.shape[-1]:], noise)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

            # 验证
            if self.val_dataloader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
                # 保存最好模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Best model saved (Val Loss = {val_loss:.4f})")

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                label, node_feature, pos_w, pos_d, r_id, adj_mx = [b.to(self.device) for b in batch]

                # label = label.unsqueeze(1)  # [B, V, T] -> [B, 1, V, T]
                # node_feature = node_feature.unsqueeze(1)

                cond = (node_feature, pos_w, pos_d)

                B = label.size(0)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
                x_t, noise = add_noise(label, t)
                pred_noise = self.model(x_t, t, cond)
                assert pred_noise[:, :, :, -label.shape[-1]:].shape == noise.shape, f"Mismatch in shape: pred_noise slice {pred_noise[:, :, :, -label.shape[-1]:].shape} vs noise {noise.shape}"

                # 计算噪声预测的 MSE
                loss = F.mse_loss(pred_noise[:, :, :, -label.shape[-1]:], noise)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_dataloader)
        self.model.train()
        return avg_loss

    @torch.no_grad()
    def sample(self, cond, shape):
        """
        cond: (node_feature, pos_w, pos_d)
        shape: label 的形状 [B, F, V, T_p]
        返回: 采样后的 x_0
        """
        self.model.eval()
        device = self.device

        # 从随机高斯开始
        x = torch.randn(shape, device=device)

        num_steps = scheduler.config.num_train_timesteps
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor, cond)

            # 使用 scheduler 封装的去噪过程
            x = scheduler.step(pred_noise, t_tensor, x).prev_sample
        return x