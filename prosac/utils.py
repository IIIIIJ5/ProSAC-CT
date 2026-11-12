"""Small helpers shared by the model, the diffusion process and the trainer."""
import math
import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    return (img + 1) * 0.5


def extract(a, t, x_shape):
    """Gather per-sample coefficients a[t] and reshape them for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1):
    """Residual / noise schedules of the residual diffusion process (they sum to `sum_scale`)."""
    x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
    if schedule == "decreased":
        x = torch.flip(x, dims=[0])
    elif schedule != "increased":
        raise ValueError(f"unknown schedule: {schedule}")
    alphas = x / (0.5 * timesteps * (timesteps + 1))
    assert math.isclose(alphas.sum().item(), 1.0, abs_tol=1e-10)
    return alphas * sum_scale
