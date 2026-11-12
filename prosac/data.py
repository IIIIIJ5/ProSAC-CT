"""Three-slice (2.5D) LDCT observations with the centre NDCT slice as target.
数据读取：每个样本包含中心 NDCT（目标）、中心 LDCT 及其前后两张相邻 LDCT 切片。

Four aligned text files per split, one image path per line:
    <split>_ndct.txt       centre NDCT slice (target)
    <split>_ldct.txt       centre LDCT slice y_c
    <split>_ldct_prev.txt  adjacent LDCT slice y_{c-1}
    <split>_ldct_next.txt  adjacent LDCT slice y_{c+1}
At the first / last slice of a volume, repeat the centre slice as its missing neighbour.
"""
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def read_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


class CTSliceDataset(Dataset):
    """Returns (NDCT (1, H, W), centre LDCT (1, H, W), adjacent LDCT (2, H, W), file name), all in [0, 1]."""

    def __init__(self, ndct_list, ldct_list, ldct_prev_list, ldct_next_list, image_size, dataroot=''):
        super().__init__()
        self.lists = [read_list(p) for p in (ndct_list, ldct_list, ldct_prev_list, ldct_next_list)]
        assert len({len(lst) for lst in self.lists}) == 1, 'all four lists must have the same length'
        self.dataroot = dataroot
        self.image_size = image_size

    def __len__(self):
        return len(self.lists[0])

    def _load(self, rel_path):
        # 补零到 8 的倍数（且不小于 image_size），再双三次插值缩放到 image_size x image_size
        img = np.asarray(Image.open(os.path.join(self.dataroot, rel_path)).convert('L'))
        img = pad_to_size(img, self.image_size)
        img = Image.fromarray(img).resize((self.image_size, self.image_size), Image.BICUBIC)
        return TF.to_tensor(np.array(img)).float()

    def __getitem__(self, index):
        ndct, ldct, prev, nxt = [self._load(lst[index]) for lst in self.lists]
        return ndct, ldct, torch.cat((prev, nxt), dim=0), os.path.basename(self.lists[0][index])


def pad_to_size(img, min_size, block_size=8):
    """Zero-pads the bottom/right so that both sides are >= min_size and multiples of block_size."""
    h, w = img.shape[:2]
    target_h = int(np.ceil(max(h, min_size) / block_size) * block_size)
    target_w = int(np.ceil(max(w, min_size) / block_size) * block_size)
    return cv2.copyMakeBorder(img, 0, target_h - h, 0, target_w - w, cv2.BORDER_CONSTANT, value=0)
