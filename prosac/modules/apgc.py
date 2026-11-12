"""Anatomical Prior-Guided Conditioning (APGC).

The prior is computed from the centre LDCT slice y_c only:
    F_AFM  = MedDINOv3(y_c)                       anatomical feature map (frozen backbone + trainable alignment)
    F_EFM  = Sob(y_c) + Lap(y_c)                  edge feature map
    F_AEFM = (alpha * F_AFM) * (gamma * F_EFM)    alpha, gamma: learnable channel-wise weights
All maps are brought to the fusion resolution and C channels by convolutional projections.
The result does not depend on the time step, so it is computed once per slice and reused at every reverse step.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .meddinov3 import ANATOMY_DIM, MedDINOv3Encoder


def sobel_laplacian_maps(x):
    """(B, 1, H, W) -> Sobel gradient magnitude and |Laplacian|, each normalised by its per-image maximum."""
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f'expected a (B, 1, H, W) tensor, got {tuple(x.shape)}')
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    sobel = torch.sqrt(F.conv2d(x, sobel_x, padding=1) ** 2 + F.conv2d(x, sobel_y, padding=1) ** 2 + 1e-8)
    lap = torch.abs(F.conv2d(x, laplacian, padding=1))
    return (sobel / (sobel.amax(dim=(2, 3), keepdim=True) + 1e-8),
            lap / (lap.amax(dim=(2, 3), keepdim=True) + 1e-8))


class APGC(nn.Module):
    def __init__(self, dim, backbone=None):
        super().__init__()
        self.anatomy_encoder = MedDINOv3Encoder(backbone=backbone)
        self.afm_proj = nn.Conv2d(ANATOMY_DIM, dim, 1)
        self.sobel_proj = nn.Conv2d(1, dim, 3, padding=1)
        self.laplacian_proj = nn.Conv2d(1, dim, 3, padding=1)
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, y_c):
        """y_c: centre LDCT slice in [0, 1], (B, 1, H, W) -> F_AEFM, (B, dim, H, W)."""
        afm = self.afm_proj(self.anatomy_encoder(y_c))
        sobel, lap = sobel_laplacian_maps(y_c)
        efm = self.sobel_proj(sobel) + self.laplacian_proj(lap)
        return (self.alpha * afm) * (self.gamma * efm)
