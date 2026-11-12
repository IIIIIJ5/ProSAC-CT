"""Conversion to 8-bit images and side-by-side result panels."""
import os

import cv2
import numpy as np
from PIL import Image


def to_uint8(tensor):
    """(C, H, W) tensor in [0, 1] -> (H, W, C) uint8 array."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.uint8(np.clip(arr, 0, 1) * 255)


def save_panel(input_img, output_img, target_img, psnr, ssim, file_name, save_dir):
    """Saves [input | output | target] with the metrics written above the panel."""
    os.makedirs(save_dir, exist_ok=True)
    combined = np.hstack((input_img, output_img, target_img))
    h, w, c = combined.shape
    text_h = 30
    panel = np.zeros((h + text_h, w, c), dtype=np.uint8)
    panel[text_h:] = combined

    text = 'psnr:{:.2f}, ssim:{:.2f}, {}'.format(int(psnr * 100) / 100.0, int(ssim * 100) / 100.0, file_name)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(panel, text, ((w - text_w) // 2, 20), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if panel.shape[2] == 1:
        panel = panel[:, :, 0]
    Image.fromarray(panel).save(os.path.join(save_dir, file_name))


def save_image(img, file_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img = img[:, :, 0] if img.ndim == 3 and img.shape[2] == 1 else img
    Image.fromarray(img).save(os.path.join(save_dir, os.path.splitext(file_name)[0] + '.png'))
