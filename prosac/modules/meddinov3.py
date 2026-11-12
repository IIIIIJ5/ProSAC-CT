"""Frozen MedDINOv3 ViT-B/16 encoder that turns an LDCT slice into multi-scale anatomical features.
冻结的 MedDINOv3 编码器：从 LDCT 切片提取解剖特征（主干冻结，仅上采样层参与训练）。

MedDINOv3 code and weights are not bundled here. See README.md ("MedDINOv3 setup").
By default the code looks for:
    <project root>/MedDINOv3/nnUNet                          (clone of github.com/ricklisz/MedDINOv3)
    <project root>/MedDINOv3/MedDINOv3-ViTB-16-CT-3M/model.pth
Both locations can be overridden with the MEDDINOV3_REPO and MEDDINOV3_CKPT environment variables.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import Upsample

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REPO = os.path.join(PROJECT_ROOT, 'MedDINOv3')
DEFAULT_CKPT = os.path.join(DEFAULT_REPO, 'MedDINOv3-ViTB-16-CT-3M', 'model.pth')


def build_meddinov3_vitb(repo_dir=None, ckpt_path=None):
    repo_dir = repo_dir or os.environ.get('MEDDINOV3_REPO', DEFAULT_REPO)
    ckpt_path = ckpt_path or os.environ.get('MEDDINOV3_CKPT', DEFAULT_CKPT)

    nnunet_path = os.path.join(repo_dir, 'nnUNet')
    dinov3_path = os.path.join(nnunet_path, 'nnunetv2', 'training', 'nnUNetTrainer', 'dinov3')
    for path in (nnunet_path, dinov3_path):  # the DINOv3 code imports itself as a top-level `dinov3` package
        if path not in sys.path:
            sys.path.insert(0, path)
    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_base

    model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05, n_storage_tokens=4,
                     qkv_bias=False, mask_k_bias=True)
    state_dict = torch.load(ckpt_path, weights_only=False, map_location='cpu')['teacher']
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()
                  if 'ibot' not in k and 'dino_head' not in k}
    model.load_state_dict(state_dict)
    return model


ANATOMY_DIM = 96  # channels of the aligned MedDINOv3 feature map


class MedDINOv3Encoder(nn.Module):
    """Frozen ViT-B/16 tokens -> trainable upsampling to 1/4 resolution -> bilinear resize to the input size.

    Returns a (B, ANATOMY_DIM, H, W) anatomical feature map.
    """

    def __init__(self, backbone=None):
        super().__init__()
        self.model = backbone if backbone is not None else build_meddinov3_vitb()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # ViT-B/16 tokens are at 1/16 resolution: upsample to 1/8 and 1/4.
        self.conv_up8 = Upsample(768, 192)
        self.conv_up4 = Upsample(192, ANATOMY_DIM)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        with torch.no_grad():
            tokens = self.model(x, is_training=True)['x_norm_patchtokens']
            b, l, c = tokens.shape
            h = int(math.sqrt(l))
            tokens = tokens.permute(0, 2, 1).reshape(b, c, h, h)

        feat = self.conv_up4(self.conv_up8(tokens))
        return F.interpolate(feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
