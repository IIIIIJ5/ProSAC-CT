"""Residual Frequency-Domain Decoupling Layer (RFDDL); RFDDS holds one RFDDL per denoising stage.

    F_b   = Re[ IFFT( ifftshift( M_b * fftshift(FFT(F_SLR)) ) ) ],   b in {low, mid, high}
    z_b   = GAP( DBR_b(F_b) )                                       (channel descriptor, R^C)
    w     = Softmax_over_bands([z_l, z_m, z_h])                     (channel-wise band weights)
    F_hat = F_SLR + eta * sum_b w_b * F_b

M_b are radial masks on the centred spectrum with cut-offs r1 and r2 of the normalised radius.
残差频域解耦层：把特征分成低/中/高三个径向频段，由频段自身估计通道级权重，加权后以残差方式加回。
"""
import torch
import torch.nn.functional as F
from torch import nn


def radial_band_masks(h, w, r1, r2, device):
    u = torch.arange(h, device=device, dtype=torch.float32) - h // 2
    v = torch.arange(w, device=device, dtype=torch.float32) - w // 2
    radius = torch.sqrt(u[:, None] ** 2 + v[None, :] ** 2) / ((h / 2) ** 2 + (w / 2) ** 2) ** 0.5
    return [(radius <= r1).float(), ((radius > r1) & (radius <= r2)).float(), (radius > r2).float()]


class RFDDL(nn.Module):
    def __init__(self, dim, r1=0.3, r2=0.7):
        super().__init__()
        self.r1, self.r2 = r1, r2
        # DBR: 3x3 depth-wise conv -> BN -> ReLU, one unit per band (no weight sharing)
        self.dbr = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim), nn.BatchNorm2d(dim), nn.ReLU())
            for _ in range(3)])
        self.eta = nn.Parameter(torch.zeros(1, dim, 1, 1))  # starts as identity mapping

    def decouple(self, x):
        """Low / mid / high frequency components of x in the spatial domain."""
        spec = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        masks = radial_band_masks(x.shape[-2], x.shape[-1], self.r1, self.r2, x.device)
        return [torch.fft.ifft2(torch.fft.ifftshift(spec * m, dim=(-2, -1)), dim=(-2, -1)).real for m in masks]

    def forward(self, x):
        bands = self.decouple(x)
        # 每个频段经 DBR 后全局平均池化，得到通道描述子；在三个频段之间做 softmax
        z = torch.stack([dbr(band).mean(dim=(2, 3)) for dbr, band in zip(self.dbr, bands)], dim=1)  # (B, 3, C)
        w = F.softmax(z, dim=1)
        response = sum(w[:, b, :, None, None] * bands[b] for b in range(3))
        return x + self.eta * response
