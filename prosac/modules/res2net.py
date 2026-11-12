"""Res2NeXt-style residual block used in the two shallow encoder levels and the two deep decoder levels.
Res2NeXt 残差块：第二个卷积换成多尺度分组分支，用在编码器浅层和解码器深层。

Same interface as the time-conditioned ResnetBlock; only the second convolution is replaced by a
multi-scale Res2Net branch with grouped (ResNeXt) convolutions and a gated merge.
"""
import torch
from einops import rearrange
from torch import nn

from ..utils import exists
from .blocks import WeightStandardizedConv2d


class ConvNormAct(nn.Module):
    """Weight-standardised conv -> GroupNorm -> optional FiLM scale/shift -> SiLU."""

    def __init__(self, dim, dim_out, kernel_size=3, padding=1, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Res2NeXtBlock(nn.Module):
    def __init__(self, dim, groups=8, cardinality=2, scale=4):
        super().__init__()
        self.scale = scale
        total_groups = groups * cardinality
        mid_dim = dim // scale

        self.reduce_conv = ConvNormAct(dim, dim, 1, 0, total_groups)
        # One branch per channel split. The first split is passed through unchanged, so branches[0] is unused;
        # it is kept so that parameter names match the trained checkpoints.
        self.branches = nn.ModuleList([
            nn.Sequential(WeightStandardizedConv2d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=total_groups),
                          nn.GroupNorm(total_groups, mid_dim),
                          nn.SiLU())
            for _ in range(scale)])
        self.gate_conv = ConvNormAct(dim, dim * 2, 1, 0, groups)
        self.merge_conv = ConvNormAct(dim, dim, 1, 0, groups)
        self.gate = SimpleGate()

    def forward(self, x):
        chunks = torch.chunk(self.reduce_conv(x), self.scale, dim=1)
        outs = [chunks[0]]
        for i in range(1, self.scale):
            outs.append(self.branches[i](outs[i - 1] + chunks[i]))
        h = self.gate(self.gate_conv(torch.cat(outs, dim=1)))
        return self.merge_conv(h)


class Res2netBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, cardinality=2, scale=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = ConvNormAct(dim, dim_out, groups=groups)
        self.block2 = Res2NeXtBlock(dim_out, groups=groups, cardinality=cardinality, scale=scale)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = rearrange(self.mlp(time_emb), 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
