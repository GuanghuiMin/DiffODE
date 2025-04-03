# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import math
import torch.nn.init as init
import numpy as np
from utils import *

"""
Implementation of UGnet with improved skip connection handling and corrected dropout parameter naming.
TcnBlock: extract time feature
SpatialBlock: extract the spatial feature
"""

def TimeEmbedding(timesteps: torch.Tensor, embedding_dim: int):
    """
    Build sinusoidal embeddings as in Denoising Diffusion Probabilistic Models.
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatialBlock, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # x: [B, c_in, time, n_nodes]
        # Lk: [3, n_nodes, n_nodes]
        if len(Lk.shape) == 2:  # if supports_len == 1:
            Lk = Lk.unsqueeze(0)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # [B, c_out, time, n_nodes]
        return torch.relu(x_gc + x)


class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, : -self.chomp_size]


class TcnBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation_size=1, dropout=0.0):  # 修正了参数名称
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = (self.kernel_size - 1) * self.dilation_size

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(3, self.kernel_size), padding=(1, self.padding), dilation=(1, self.dilation_size))
        self.chomp = Chomp(self.padding)
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.chomp, self.drop)
        self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None

    def forward(self, x):
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)
        return out + x_skip


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, config, kernel_size=3):
        """
        TCN convolution block with skip connection.
        """
        super().__init__()
        self.tcn1 = TcnBlock(c_in, c_out, kernel_size=kernel_size)
        self.tcn2 = TcnBlock(c_out, c_out, kernel_size=kernel_size)
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1,1))
        self.t_conv = nn.Conv2d(config.d_h, c_out, (1,1))
        self.spatial = SpatialBlock(config.supports_len, c_out, c_out)
        self.norm = nn.LayerNorm([config.V, c_out])

    def forward(self, x, t, A_hat):
        h = self.tcn1(x)
        h += self.t_conv(t[:, :, None, None])
        h = self.tcn2(h)
        h = self.norm(h.transpose(1,3)).transpose(1,3)  # (B, c_out, V, T)
        h = h.transpose(2,3)  # (B, c_out, T, V)
        h = self.spatial(h, A_hat).transpose(2,3)  # (B, c_out, V, T)
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        """
        DownBlock: 一个残差块，输出尺寸 (B, c_out, V, T)
        """
        super().__init__()
        self.res = ResidualBlock(c_in, c_out, config, kernel_size=3)

    def forward(self, x, t, supports):
        return self.res(x, t, supports)


class Downsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in,  kernel_size=(1,3), stride=(1,2), padding=(0,1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, supports):
        return self.conv(x)


class DownBlockStack(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    def forward(self, x, t, supports):
        for block in self.blocks:
            x = block(x, t, supports)
        return x


class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        """
        UpBlock：在解码器中，输入是拼接后的特征 (低分辨率特征 + 编码器跳跃连接)，
        输出为 c_out 通道的特征。
        """
        super().__init__()
        self.res = ResidualBlock(c_in + c_out, c_out, config, kernel_size=3)

    def forward(self, x, t, supports):
        return self.res(x, t, supports)


class Upsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_in, (1, 4), (1, 2), (0, 1))

    def forward(self, x, t, supports):
        return self.conv(x)


class MiddleBlock(nn.Module):
    def __init__(self, c_in, config):
        super().__init__()
        self.res1 = ResidualBlock(c_in, c_in, config, kernel_size=3)
        self.res2 = ResidualBlock(c_in, c_in, config, kernel_size=3)

    def forward(self, x, t, supports):
        x = self.res1(x, t, supports)
        x = self.res2(x, t, supports)
        return x


class UGnet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.d_h = config.d_h
        self.T_p = config.T_p
        self.T_h = config.T_h
        T = self.T_p + self.T_h
        self.F = config.F

        self.n_blocks = config.n_blocks#config.get('n_blocks', 2)
        n_resolutions = len(config.channel_multipliers)

        # 构造编码器（down-phase）：每一层由 n_blocks 个 DownBlock 组成，且除最后一层外，在层末加 Downsample
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        encoder_channels = []
        in_channels = self.d_h
        for i in range(n_resolutions):
            out_channels = in_channels * config.channel_multipliers[i]
            block = DownBlockStack([DownBlock(in_channels, out_channels, config) for _ in range(self.n_blocks)])
            self.down_blocks.append(block)
            encoder_channels.append(out_channels)
            in_channels = out_channels
            if i < n_resolutions - 1:
                self.downsamples.append(Downsample(in_channels))

        self.middle = MiddleBlock(in_channels, config)

        # 构造解码器（up-phase）：倒序使用编码器中各层的跳跃连接，并在各层前（除第一层外）进行上采样
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current_channels = in_channels
        for i in reversed(range(n_resolutions)):
            if i < n_resolutions - 1:
                self.upsamples.append(Upsample(current_channels))
            # UpBlock：输入拼接后的通道数为 current_channels + encoder_channels[i]，输出 encoder_channels[i]
            self.up_blocks.append(UpBlock(current_channels, encoder_channels[i], config))
            current_channels = encoder_channels[i]

        # 投影层
        self.x_proj = nn.Conv2d(self.F, self.d_h, (1,1))
        self.out = nn.Sequential(
            nn.Conv2d(self.d_h, self.F, (1,1)),
            nn.Linear(2 * T, T),
        )
        # 构造图卷积的邻接矩阵（GCN部分）
        a1 = asym_adj(config.A)
        a2 = asym_adj(np.transpose(config.A))
        self.a1 = torch.from_numpy(a1).to(config.device)
        self.a2 = torch.from_numpy(a2).to(config.device)
        config.supports_len = 2

    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        """
        x: (B, F, V, T)
        t: diffusion step
        c: tuple, 包含 (x_masked, pos_w, pos_d)
        """
        x_masked, pos_w, pos_d = c  # pos_w 和 pos_d 目前未使用
        # 拼接后时间维度变为 2 * T
        x = torch.cat((x, x_masked), dim=3)  # (B, F, V, 2*T)
        x = self.x_proj(x)  # (B, d_h, V, 2*T)
        t = TimeEmbedding(t, self.d_h)
        supports = torch.stack([self.a1, self.a2])

        skip_connections = []
        # 编码器
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x, t, supports)
            skip_connections.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x, t, supports)
        x = self.middle(x, t, supports)

        # 解码器：反向使用跳跃连接
        num_up = len(self.up_blocks)
        for i in range(num_up):
            if i > 0:
                x = self.upsamples[i-1](x, t, supports)
            # 使用与当前分辨率对应的跳跃连接
            skip = skip_connections[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = self.up_blocks[i](x, t, supports)
        e = self.out(x)
        return e