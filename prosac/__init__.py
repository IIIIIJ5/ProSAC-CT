from .diffusion import ResidualDiffusion
from .unet import ProSACUNet

__all__ = ['ResidualDiffusion', 'ProSACUNet', 'build_prosac_ct']


def build_prosac_ct(image_size=128, dim=64, dim_mults=(1, 2, 4, 8), sampling_timesteps=10, use_apgc=True,
                    use_rfdds=True, use_td3=True, loss_weights=(1.0, 1.0, 0.1), ct_backbone=None):
    """ProSAC-CT: APGC + shared encoder + RFDDS + TD3, residual diffusion with the stage-wise L2 loss.

    use_apgc / use_rfdds / use_td3 are the module switches (默认全部开启，关闭即为对应消融).
    """
    unet = ProSACUNet(dim=dim, dim_mults=dim_mults, channels=1, adjacent_slices=2, use_apgc=use_apgc,
                      use_rfdds=use_rfdds, use_td3=use_td3, ct_backbone=ct_backbone)
    return ResidualDiffusion(unet, image_size=image_size, timesteps=1000, sampling_timesteps=sampling_timesteps,
                             sum_scale=0.01, loss_weights=loss_weights)
