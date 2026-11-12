"""PSNR / SSIM on 8-bit images (same definitions as BasicSR).
评价指标：在 0-255 的 8 位图像上计算 PSNR 与 SSIM。
"""
import cv2
import numpy as np


def _to_hwc_float(img):
    img = img.astype(np.float64)
    return img[..., None] if img.ndim == 2 else img


def calculate_psnr(img, img2):
    """img, img2: uint8 arrays (H, W) or (H, W, C) in [0, 255]."""
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    mse = np.mean((_to_hwc_float(img) - _to_hwc_float(img2)) ** 2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)


def calculate_ssim(img, img2):
    """SSIM with an 11x11 Gaussian window (sigma 1.5), averaged over channels."""
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    img, img2 = _to_hwc_float(img), _to_hwc_float(img2)
    return float(np.mean([_ssim(img[..., i], img2[..., i]) for i in range(img.shape[2])]))


def _ssim(img, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid region of the 11x11 window
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()
