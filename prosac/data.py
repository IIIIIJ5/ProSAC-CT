"""Three-slice (2.5D) LDCT observations with the centre NDCT slice as target.

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
    """Returns (NDCT (1, H, W), centre LDCT (1, H, W), adjacent LDCT (2, H, W), file name), all in [0, 1].

    Each slice is zero-padded on the bottom/right to at least `image_size` and to a multiple of 8,
    then resized to image_size x image_size with bicubic interpolation.
    """

    def __init__(self, ndct_list, ldct_list, ldct_prev_list, ldct_next_list, image_size, dataroot='',
                 augment_flip=False):
        super().__init__()
        self.lists = [read_list(p) for p in (ndct_list, ldct_list, ldct_prev_list, ldct_next_list)]
        assert len({len(lst) for lst in self.lists}) == 1, 'all four lists must have the same length'
        self.dataroot = dataroot
        self.image_size = image_size
        self.augment_flip = augment_flip

    def __len__(self):
        return len(self.lists[0])

    def _load(self, rel_path):
        img = np.asarray(Image.open(os.path.join(self.dataroot, rel_path)).convert('L'))
        img = pad_to_size(img, self.image_size)
        img = Image.fromarray(img).resize((self.image_size, self.image_size), Image.BICUBIC)
        return np.asarray(img)

    def __getitem__(self, index):
        images = [self._load(lst[index]) for lst in self.lists]
        if self.augment_flip:
            images = [np.ascontiguousarray(np.fliplr(img)) for img in images]
        ndct, ldct, prev, nxt = [TF.to_tensor(img).float() for img in images]
        return ndct, ldct, torch.cat((prev, nxt), dim=0), os.path.basename(self.lists[0][index])


def pad_to_size(img, min_size, block_size=8):
    h, w = img.shape[:2]
    target_h = int(np.ceil(max(h, min_size) / block_size) * block_size)
    target_w = int(np.ceil(max(w, min_size) / block_size) * block_size)
    return cv2.copyMakeBorder(img, 0, target_h - h, 0, target_w - w, cv2.BORDER_CONSTANT, value=0)
