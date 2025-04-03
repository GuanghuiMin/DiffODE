# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np
from torch.utils.data import Subset

# Import the new dataset classes that return (label, node_feature, pos_w, pos_d, r_id, adj_mx)
from dataset import InfectionGraphDataset, InfectionTransformedDataset
# Import the new UGnet model
from ugnet import UGnet
# Import the DiffusionTrainer and scheduler
from diffusion_trainer import DiffusionTrainer, scheduler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Spatio-Temporal Prediction Training")
    parser.add_argument("--seed", type=int, default=8, help="Random seed")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to the pickle data file")
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size for quick testing")
    parser.add_argument("--T_h", type=int, default=7, help="History steps")
    parser.add_argument("--T_p", type=int, default=7, help="Prediction steps")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    # UGnet parameters
    parser.add_argument("--d_h", type=int, default=64, help="Dimension for hidden channels")
    parser.add_argument("--F", type=int, default=1, help="Input feature dimension (channel)")
    parser.add_argument("--channel_multipliers", type=int, nargs="+", default=[2, 2, 2], help="Channel multipliers")
    parser.add_argument("--n_blocks", type=int, default=2, help="Number of blocks per resolution")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    setup_seed(args.seed)

    # 1) Load the base dataset. InfectionGraphDataset returns the entire series and the global stats.
    base_dataset = InfectionGraphDataset(pkl_path=args.pkl_path, use_global_stats=True)

    # 2) Create the transformed dataset which returns 6 elements: (label, node_feature, pos_w, pos_d, r_id, adj_mx)
    transformed_dataset = InfectionTransformedDataset(base_dataset, T_h=args.T_h, T_p=args.T_p)

    # 3) Split the dataset into training and validation sets
    n = len(transformed_dataset)
    n_train = int(0.8 * n)
    train_dataset = Subset(transformed_dataset, range(n_train))
    val_dataset = Subset(transformed_dataset, range(n_train, n))

    # 4) Configure UGnet
    class Config:
        pass

    config = Config()
    config.device = args.device
    config.d_h = args.d_h
    config.F = args.F
    config.channel_multipliers = args.channel_multipliers
    config.n_blocks = args.n_blocks
    config.T_h = args.T_h
    config.T_p = args.T_p
    config.V = base_dataset.samples.shape[1]
    config.A = base_dataset.adj_mx.numpy()[:config.V, :config.V]
    config.supports_len = 2
    print(base_dataset.samples.shape)

    # 5) Instantiate UGnet and move it to the specified device
    model = UGnet(config).to(args.device)

    # 6) Create the DiffusionTrainer with the model and datasets
    trainer = DiffusionTrainer(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device
    )

    # 7) Train the model
    trainer.train(num_epochs=args.num_epochs, save_path="best_model.pt")

    # 8) After training, perform a sample generation from a validation sample
    sample_index = 0
    label, node_feature, pos_w, pos_d, r_id, adj_mx = val_dataset[sample_index]

    # Ensure tensors are of type torch.Tensor
    if not torch.is_tensor(label):
        label = torch.tensor(label)
    if not torch.is_tensor(node_feature):
        node_feature = torch.tensor(node_feature)
    if not torch.is_tensor(pos_w):
        pos_w = torch.tensor(pos_w)
    if not torch.is_tensor(pos_d):
        pos_d = torch.tensor(pos_d)

    # Construct condition: (node_feature, pos_w, pos_d)
    cond = (node_feature, pos_w, pos_d)

    # The shape for sampling should be (B, F, V, T_p); label is [1,1,T_p] so we assume F=1, V=1
    shape = (1,) + label.shape  # results in (1, 1, 1, T_p)

    generated = trainer.sample(cond, shape)
    print("Sampled output:", generated)


if __name__ == '__main__':
    main()