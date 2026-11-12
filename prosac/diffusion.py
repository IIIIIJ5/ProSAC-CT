"""Residual diffusion (RDDM) for LDCT denoising with time-step-decoupled training.

With the centre-slice residual r = y_c - x_0 (x_0 = NDCT):
    x_t    = x_0 + A_t r + B_t eps,            A_t = sum_{i<=t} a_i,  B_t = sqrt(sum_{i<=t} b_i^2)
    x0_hat = x_t - A_t r_hat - B_t eps_hat
Stage-dependent loss (all stages use (lambda_r, lambda_eps, lambda_x) = (1, 1, 0.1)):
    L = lambda_r ||r - r_hat||^2 + lambda_eps ||eps - eps_hat||^2 + lambda_x ||x0_hat - x_0||^2
Deterministic residual sampler between selected steps t > t':
    x_t' = x_t - (A_t - A_t') r_hat - (B_t - B_t') eps_hat
With 10 sampling steps the network is evaluated at t = 999, 899, ..., 99 and the last x0_hat is returned.
残差扩散：中心 LDCT 与 NDCT 的差作为残差，网络同时预测残差和噪声；推理用确定性的残差采样器。
"""
import math
import random

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .unet import NUM_STAGES, STAGE_RANGES
from .utils import extract, gen_coefficients, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one


class ResidualDiffusion(nn.Module):
    def __init__(self, model, *, image_size, timesteps=1000, sampling_timesteps=10, sum_scale=0.01,
                 loss_weights=(1.0, 1.0, 0.1)):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        assert sampling_timesteps <= timesteps
        self.sum_scale = sum_scale
        self.lambda_r, self.lambda_eps, self.lambda_x = loss_weights

        alphas_cumsum = gen_coefficients(timesteps, schedule="decreased").cumsum(dim=0).clip(0, 1)
        betas2_cumsum = gen_coefficients(timesteps, schedule="increased", sum_scale=sum_scale).cumsum(dim=0).clip(0, 1)
        self.register_buffer('alphas_cumsum', alphas_cumsum.to(torch.float32))           # A_t
        self.register_buffer('betas2_cumsum', betas2_cumsum.to(torch.float32))
        self.register_buffer('betas_cumsum', torch.sqrt(betas2_cumsum).to(torch.float32))  # B_t

    def q_sample(self, x0, x_res, t, noise):
        # 前向过程：x_t = x0 + A_t * r + B_t * noise
        return x0 + extract(self.alphas_cumsum, t, x0.shape) * x_res + extract(self.betas_cumsum, t, x0.shape) * noise

    def predict_x0(self, x_t, t, pred_res, pred_noise):
        # 由预测的残差和噪声恢复 NDCT：x0_hat = x_t - A_t * r_hat - B_t * eps_hat
        return x_t - extract(self.alphas_cumsum, t, x_t.shape) * pred_res - extract(self.betas_cumsum, t, x_t.shape) * pred_noise

    # ----------------------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, y_c, y_adj):
        """y_c: centre LDCT (B, 1, H, W); y_adj: adjacent LDCT slices (B, 2, H, W); both in [0, 1].

        Returns the NDCT estimate in [0, 1].
        """
        prior = self.model.encode_prior(y_c)            # computed once, reused at every reverse step
        y_c, y_adj = normalize_to_neg_one_to_one(y_c), normalize_to_neg_one_to_one(y_adj)

        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        img = y_c + math.sqrt(self.sum_scale) * torch.randn_like(y_c)  # 从加少量噪声的中心 LDCT 出发
        for time, time_next in tqdm(list(zip(times[:-1], times[1:])), desc='sampling', leave=False):
            t = torch.full((img.shape[0],), time, device=img.device, dtype=torch.long)
            pred_res, pred_noise = self.model(torch.cat((img, y_adj), dim=1), t, prior)
            pred_res = pred_res.clamp(-1., 1.)
            if time_next < 0:
                img = self.predict_x0(img, t, pred_res, pred_noise).clamp(-1., 1.)
                break
            img = (img - (self.alphas_cumsum[time] - self.alphas_cumsum[time_next]) * pred_res
                   - (self.betas_cumsum[time] - self.betas_cumsum[time_next]) * pred_noise)
        return unnormalize_to_zero_to_one(img)

    # ----------------------------------------------------------------------------- training
    def sample_stage_timesteps(self, batch_size, device):
        """With TD3: one stage per mini-batch, then a uniform time step inside that stage for every sample.
        TD3 开启时一个 batch 只训练一个阶段的分支；关闭时在全部时间步上均匀采样。
        """
        if not self.model.use_td3:
            return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        lo, hi = STAGE_RANGES[random.randint(0, NUM_STAGES - 1)]
        return torch.randint(lo, hi + 1, (batch_size,), device=device)

    def forward(self, x_ndct, y_c, y_adj):
        """All inputs in [0, 1]: NDCT (B, 1, H, W), centre LDCT (B, 1, H, W), adjacent LDCT (B, 2, H, W)."""
        prior = self.model.encode_prior(y_c)
        x0 = normalize_to_neg_one_to_one(x_ndct)
        y_c, y_adj = normalize_to_neg_one_to_one(y_c), normalize_to_neg_one_to_one(y_adj)

        t = self.sample_stage_timesteps(x0.shape[0], x0.device)
        noise = torch.randn_like(x0)
        x_res = y_c - x0
        x_t = self.q_sample(x0, x_res, t, noise)

        pred_res, pred_noise = self.model(torch.cat((x_t, y_adj), dim=1), t, prior)
        x0_hat = self.predict_x0(x_t, t, pred_res, pred_noise)
        # 分阶段损失：残差、噪声、重建 NDCT 三项 L2，权重 (1, 1, 0.1)
        return (self.lambda_r * F.mse_loss(pred_res, x_res) + self.lambda_eps * F.mse_loss(pred_noise, noise)
                + self.lambda_x * F.mse_loss(x0_hat, x0))
