import torch
import math
import pickle
import numpy as np
from torch.utils.data import Dataset

class InfectionGraphDataset(Dataset):
    """
    以日频方式：
      - 使用全局 mean-std 做标准化
      - 使用统一的 [T, V] 感染张量表示所有区域
    """

    def __init__(self, pkl_path, use_global_stats=True):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.region2idx = {}
        idx_counter = 0

        # 1) 建立 region 映射
        for region in self.data.keys():
            if region not in self.region2idx:
                self.region2idx[region] = idx_counter
                idx_counter += 1

        self.num_regions = len(self.region2idx)

        # 获取所有区域中最短的 daily_cases 长度
        min_len = min(len(entry["daily_cases"]) for entry in self.data.values())

        # 初始化统一长度的 raw_cases 矩阵
        self.raw_cases = np.zeros((min_len, self.num_regions), dtype=np.float32)
        for region, entry in self.data.items():
            region_idx = self.region2idx[region]
            daily_cases = entry["daily_cases"][:min_len]  # 截断
            daily_cases = [0.0 if math.isnan(v) else v for v in daily_cases]
            self.raw_cases[:, region_idx] = daily_cases

        # 3) 计算 mean, std
        if use_global_stats:
            self.global_mean = float(np.mean(self.raw_cases))
            self.global_std  = float(np.std(self.raw_cases) + 1e-6)
        else:
            self.global_mean = 0.0
            self.global_std  = 1.0

        # 4) 生成 samples
        self.samples = (self.raw_cases - self.global_mean) / self.global_std

        # 5) 建立邻接矩阵
        self.adj_mx = torch.zeros((self.num_regions, self.num_regions), dtype=torch.float32)
        for region, entry in self.data.items():
            region_idx = self.region2idx[region]
            for neighbor in entry["neighbors"]:
                neighbor_idx = self.region2idx[neighbor]
                self.adj_mx[region_idx, neighbor_idx] = 1.0

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        cases_norm = torch.tensor(self.samples[idx], dtype=torch.float32)
        return cases_norm, self.adj_mx

def search_recent_data(infection_tensor, length, label_start_idx, T_h, T_p):
    if label_start_idx + T_p > length:
        return None
    hist_start = label_start_idx - T_h
    hist_end   = label_start_idx
    if hist_start < 0:
        return None
    return (hist_start, hist_end), (label_start_idx, label_start_idx + T_p)

class InfectionTransformedDataset(Dataset):
    """
    读取 DailyInfectionGraphDataset（标准化后），
    切分全图的时间序列为 (T_h, T_p) 窗口。
    输出 (label, node_feature, pos_w, pos_d, r_id, adj_mx)。

    其中 label, node_feature 最终会被扩展为 4D 张量，
    label: [1, 1, V, T_p]，node_feature: [1, 1, V, T_h]，
    V 为区域数，全图信息，用于图扩散。
    """
    def __init__(self, base_dataset, T_h=7, T_p=7):
        self.T_h = T_h
        self.T_p = T_p

        # Use base_dataset.samples directly, which is a numpy array of shape [T_total, V]
        T_total = base_dataset.samples.shape[0]
        self.num_regions = base_dataset.samples.shape[1]
        self.full_series = base_dataset.samples  # already normalized, shape [T_total, V]
        self.adj_mx = base_dataset.adj_mx

        # Generate time window index list for each sample
        self.idx_lst = []
        # Sample from T_h to T_total - T_p inclusive
        for label_start_idx in range(self.T_h, T_total - self.T_p + 1):
            h_start = label_start_idx - self.T_h
            h_end = label_start_idx
            l_start = label_start_idx
            l_end = label_start_idx + self.T_p
            self.idx_lst.append((h_start, h_end, l_start, l_end))

    def __len__(self):
        return len(self.idx_lst)

    def __getitem__(self, index):
        h_start, h_end, l_start, l_end = self.idx_lst[index]
        # 从 full_series 中取出历史和预测窗口，full_series: [T_total, V]
        hist_window = self.full_series[h_start:h_end, :]   # [T_h, V]
        label_window = self.full_series[l_start:l_end, :]    # [T_p, V]

        # 转置，使维度变为 [V, T]
        hist_window = hist_window.transpose(0, 1)   # [V, T_h]
        label_window = label_window.transpose(0, 1) # [V, T_p]

        # 转换为 torch.Tensor，并增加 batch 与 channel 维度
        # 最终希望 label: [1, 1, V, T_p], node_feature: [1, 1, V, T_h]
        node_feature = torch.from_numpy(hist_window).unsqueeze(0)  # shape: [1, V, T_h]
        label = torch.from_numpy(label_window).unsqueeze(0)           # shape: [1, V, T_p]

        # 时间位置编码，使用 label 窗口对应的时间索引
        idx_arr = np.arange(l_start, l_end)
        pos_w = torch.from_numpy(idx_arr % 7).long()
        pos_d = torch.from_numpy(idx_arr).long()

        # 返回 r_id 为全图区域的索引
        r_id = torch.arange(self.num_regions)

        return label, node_feature, pos_w, pos_d, r_id, self.adj_mx