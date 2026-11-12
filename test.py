"""Evaluate a trained ProSAC-CT checkpoint on the test lists.

Example:
    python test.py --data_dir data/lists --checkpoint results/prosac_ct/model-100000.pt --save_dir results/test

Reports mean PSNR / SSIM against NDCT and, with --save_dir, writes the denoised slices
(<save_dir>/output) and [LDCT | denoised | NDCT] panels (<save_dir>/panel).
"""
import argparse
import os

from prosac import build_prosac_ct
from prosac.data import CTSliceDataset
from prosac.trainer import Trainer
from prosac.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--dataroot', default='')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--save_dir', default=None)
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--sampling_steps', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--no_apgc', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    test_lists = [os.path.join(args.data_dir, f'test_{name}.txt') for name in ('ndct', 'ldct', 'ldct_prev', 'ldct_next')]
    test_set = CTSliceDataset(*test_lists, image_size=args.image_size, dataroot=args.dataroot)

    diffusion = build_prosac_ct(image_size=args.image_size, sampling_timesteps=args.sampling_steps,
                                use_apgc=not args.no_apgc)
    trainer = Trainer(diffusion, test_set, test_set, results_folder=os.path.dirname(args.checkpoint) or '.',
                      eval_batch_size=args.batch_size, num_workers=0)
    trainer.load(args.checkpoint, load_optimizer=False)
    trainer.evaluate(save_dir=args.save_dir)


if __name__ == '__main__':
    main()
