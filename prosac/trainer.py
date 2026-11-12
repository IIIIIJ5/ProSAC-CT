"""Training loop with EMA, periodic evaluation (PSNR / SSIM) and checkpointing.
训练循环：EMA 权重、定期在测试集上评估、保存与恢复 checkpoint。
"""
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import calculate_psnr, calculate_ssim
from .utils import cycle, exists
from .visualize import save_image, save_panel, to_uint8


class Trainer:
    def __init__(self, diffusion, train_set, test_set, *, results_folder, batch_size=4,
                 gradient_accumulate_every=1, lr=2e-4, adam_betas=(0.9, 0.99), num_steps=100000,
                 ema_decay=0.995, ema_update_every=10, eval_every=5000, save_last_every=5000,
                 save_every=100000, eval_batch_size=64, num_workers=4, fp16=False):
        # Without CUDA, run on CPU: the complex-valued FFT ops used by the model are not supported on Apple MPS.
        self.accelerator = Accelerator(split_batches=True, mixed_precision='fp16' if fp16 else 'no',
                                       cpu=not torch.cuda.is_available())
        self.device = self.accelerator.device

        self.gradient_accumulate_every = gradient_accumulate_every
        self.num_steps = num_steps
        self.eval_every = eval_every
        self.save_last_every = save_last_every
        self.save_every = save_every
        self.eval_batch_size = eval_batch_size
        self.test_set = test_set

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
        self.dl = cycle(self.accelerator.prepare(train_loader))

        self.opt = Adam(diffusion.parameters(), lr=lr, betas=adam_betas)
        if self.accelerator.is_main_process:
            # 评估和测试都使用 EMA 权重
            self.ema = EMA(diffusion, beta=ema_decay, update_every=ema_update_every)
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(parents=True, exist_ok=True)

        self.model, self.opt = self.accelerator.prepare(diffusion, self.opt)
        self.step = 0

    # ------------------------------------------------------------------ checkpoints
    def save(self, name):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        path = self.results_folder / name
        torch.save(data, str(path))
        print(f'saved checkpoint: {path}')

    def load(self, path, load_optimizer=True):
        data = torch.load(str(path), map_location=self.device, weights_only=True)  # 只加载张量与基本类型
        self.accelerator.unwrap_model(self.model).load_state_dict(data['model'])
        self.ema.load_state_dict(data['ema'])
        self.ema.to(self.device)
        self.step = data['step']
        if load_optimizer:
            self.opt.load_state_dict(data['opt'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])
        print(f'loaded checkpoint: {path} (step {data["step"]})')

    # ------------------------------------------------------------------ training
    def train(self):
        accelerator = self.accelerator
        with tqdm(initial=self.step, total=self.num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):  # 梯度累积
                    ndct, ldct, ldct_adj, _ = next(self.dl)
                    with accelerator.autocast():
                        loss = self.model(ndct.to(self.device), ldct.to(self.device), ldct_adj.to(self.device))
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                accelerator.wait_for_everyone()
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()
                    if self.step % self.eval_every == 0:
                        self.evaluate()
                    if self.step % self.save_last_every == 0:
                        self.save('last.pt')
                    if self.step % self.save_every == 0:
                        self.save(f'model-{self.step}.pt')

                pbar.set_description(f'loss: {total_loss:.4f}')
                pbar.update(1)
        accelerator.print('training complete')

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, save_dir=None):
        """Denoises the whole test set with the EMA model and reports mean PSNR / SSIM against NDCT.

        If `save_dir` is given, saves the denoised slices to <save_dir>/output and
        [LDCT | denoised | NDCT] panels to <save_dir>/panel.
        """
        model = self.ema.ema_model
        model.eval()
        loader = DataLoader(self.test_set, batch_size=self.eval_batch_size)
        psnrs, ssims = [], []
        for ndct, ldct, ldct_adj, names in tqdm(loader, desc='evaluating'):
            output = model.sample(ldct.to(self.device), ldct_adj.to(self.device))
            for i, name in enumerate(names):
                # 在 8 位图像上计算 PSNR / SSIM
                out_img, ndct_img, ldct_img = to_uint8(output[i]), to_uint8(ndct[i]), to_uint8(ldct[i])
                psnr, ssim = calculate_psnr(out_img, ndct_img), calculate_ssim(out_img, ndct_img)
                psnrs.append(psnr)
                ssims.append(ssim)
                if save_dir is not None:
                    save_image(out_img, name, f'{save_dir}/output')
                    save_panel(ldct_img, out_img, ndct_img, psnr, ssim, name, f'{save_dir}/panel')
        psnr, ssim = float(np.mean(psnrs)), float(np.mean(ssims))
        print(f'step {self.step}: PSNR {psnr:.4f}, SSIM {ssim:.4f} on {len(psnrs)} test slices')
        return psnr, ssim
