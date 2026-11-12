"""Denoising network of ProSAC-CT for low-dose CT (LDCT) denoising.

    shared encoder( x_t, (y_{c-1}, y_{c+1}), F_AEFM )  ->  F_SLR
    RFDDL_{s(t)}( F_SLR )                               ->  F_hat_{s(t)}
    decoder_{s(t)}( F_hat_{s(t)} )                      ->  (r_hat, eps_hat)

Inputs of one call:
    x      (B, 3, H, W)    current diffusion state x_t and the two adjacent LDCT slices (all in [-1, 1])
    time   (B,)            diffusion time step (all samples of a batch share the same stage)
    prior  (B, dim, H, W)  F_AEFM from APGC, computed from the centre LDCT slice (see `encode_prior`)

Components:
    APGC  - anatomical prior-guided conditioning, injected into the shared encoder (modules/apgc.py).
    RFDDS - three residual frequency-domain decoupling layers on the shared latent representation, one per stage.
    TD3   - time-step-decoupled denoising decoder: one decoder per time-step stage.
"""
import torch
from torch import nn

from .modules.apgc import APGC
from .modules.blocks import (Attention, Downsample, LinearAttention, PreNorm, Residual, ResnetBlock,
                             SinusoidalPosEmb, Upsample)
from .modules.res2net import Res2netBlock
from .modules.rfdds import RFDDL

# Stage boundaries on the 1000-step time axis: late [0, 0.3T], middle (0.3T, 0.7T], early (0.7T, T].
NUM_TIMESTEPS = 1000
T_MID, T_LATE = 300, 700
STAGE_RANGES = [(0, T_MID), (T_MID + 1, T_LATE), (T_LATE + 1, NUM_TIMESTEPS - 1)]  # inclusive; 0 = late
NUM_STAGES = len(STAGE_RANGES)


def stage_of(t):
    """Stage index of time step t: 0 = late (low noise), 1 = middle, 2 = early (high noise)."""
    t = int(t)
    return 0 if t <= T_MID else (1 if t <= T_LATE else 2)


class EncoderStage(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim, groups, block, is_last):
        super().__init__()
        self.block1 = block(dim_in, dim_in, time_emb_dim=time_dim, groups=groups)
        self.block2 = block(dim_in, dim_in, time_emb_dim=time_dim, groups=groups)
        self.attn = Residual(PreNorm(dim_in, LinearAttention(dim_in)))
        self.downsample = Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)

    def forward(self, x, t, skips):
        x = self.block1(x, t)
        skips.append(x)
        x = self.attn(self.block2(x, t))
        skips.append(x)
        return self.downsample(x)


class DecoderStage(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim, groups, block, is_last):
        super().__init__()
        self.block1 = block(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=groups)
        self.block2 = block(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=groups)
        self.attn = Residual(PreNorm(dim_out, LinearAttention(dim_out)))
        self.upsample = Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)

    def forward(self, x, t, skips):
        x = self.block1(torch.cat((x, skips.pop()), dim=1), t)
        x = self.block2(torch.cat((x, skips.pop()), dim=1), t)
        return self.upsample(self.attn(x))


class ProSACUNet(nn.Module):
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8), channels=1, adjacent_slices=2, resnet_block_groups=8,
                 use_apgc=True, ct_backbone=None, band_cutoffs=(0.3, 0.7)):
        super().__init__()
        self.channels = channels
        groups = resnet_block_groups
        # encoder input: x_t and the adjacent LDCT slices; the centre slice enters through APGC
        self.init_conv = nn.Conv2d(channels * (1 + adjacent_slices), dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_levels = len(in_out)
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))

        self.use_apgc = use_apgc
        if use_apgc:
            self.apgc = APGC(dim, backbone=ct_backbone)

        # shared encoder: Res2Net blocks at the two shallow levels, plain residual blocks at the deep levels
        self.downs = nn.ModuleList([
            EncoderStage(dim_in, dim_out, time_dim, groups, block=Res2netBlock if level < 2 else ResnetBlock,
                         is_last=level == num_levels - 1)
            for level, (dim_in, dim_out) in enumerate(in_out)])
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=groups)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=groups)

        # RFDDS: one RFDDL per stage on the shared latent representation
        self.rfdds = nn.ModuleList([RFDDL(mid_dim, *band_cutoffs) for _ in range(NUM_STAGES)])

        # TD3: one decoder per stage (Res2Net blocks at the two deep levels, plain residual blocks above)
        self.ups_multi = nn.ModuleList([
            nn.ModuleList([
                DecoderStage(dim_in, dim_out, time_dim, groups, block=Res2netBlock if ind < 2 else ResnetBlock,
                             is_last=ind == num_levels - 1)
                for ind, (dim_in, dim_out) in enumerate(reversed(in_out))])
            for _ in range(NUM_STAGES)])

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim, groups=groups)
        self.final_conv = nn.Conv2d(dim, channels * 2, 1)  # residual and noise

    def encode_prior(self, y_c):
        """F_AEFM from the centre LDCT slice in [0, 1]; None when APGC is disabled."""
        return self.apgc(y_c) if self.use_apgc else None

    def forward(self, x, time, prior=None):
        stage = stage_of(time[0])

        x = self.init_conv(x)
        if prior is not None:
            x = x + prior
        r = x.clone()
        t = self.time_mlp(time)

        skips = []
        for level in self.downs:
            x = level(x, t, skips)
        slr = self.mid_block2(self.mid_attn(self.mid_block1(x, t)), t)

        x = self.rfdds[stage](slr)
        for level in self.ups_multi[stage]:
            x = level(x, t, skips)

        x = self.final_res_block(torch.cat((x, r), dim=1), t)
        pred_res, pred_noise = self.final_conv(x).chunk(2, dim=1)
        return pred_res, pred_noise
