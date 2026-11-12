from .diffusion import ResidualDiffusion
from .unet import ProSACUNet

__all__ = ['ResidualDiffusion', 'ProSACUNet', 'build_prosac_ct']


def build_prosac_ct(image_size=128, dim=64, dim_mults=(1, 2, 4, 8), sampling_timesteps=10, use_apgc=True,
                    ct_backbone=None):
    """Full ProSAC-CT model: APGC + shared encoder + RFDDS + TD3, residual diffusion with the stage-wise L2 loss."""
    unet = ProSACUNet(dim=dim, dim_mults=dim_mults, channels=1, adjacent_slices=2, use_apgc=use_apgc,
                      ct_backbone=ct_backbone)
    return ResidualDiffusion(unet, image_size=image_size, timesteps=1000, sampling_timesteps=sampling_timesteps,
                             sum_scale=0.01, loss_weights=(1.0, 1.0, 0.1))
