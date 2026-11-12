"""Train ProSAC-CT with the settings in config.py.

    python train.py                                   # use config.py as is
    python train.py --data_dir data/lists --debug     # override the data folder, run a 200-step check

Training resumes automatically from <results_dir>/last.pt if it exists.
训练会自动从 <results_dir>/last.pt 断点续训。
"""
import argparse
import os

import config as cfg
from prosac import build_prosac_ct
from prosac.data import CTSliceDataset
from prosac.trainer import Trainer
from prosac.utils import set_seed


def build_from_config():
    """Model with the module switches of config.py (also used by test.py)."""
    return build_prosac_ct(image_size=cfg.image_size, sampling_timesteps=cfg.sampling_steps,
                           use_apgc=cfg.use_apgc, use_rfdds=cfg.use_rfdds, use_td3=cfg.use_td3,
                           loss_weights=cfg.loss_weights)


def dataset(data_dir, split):
    lists = [os.path.join(data_dir, f'{split}_{name}.txt') for name in ('ndct', 'ldct', 'ldct_prev', 'ldct_next')]
    return CTSliceDataset(*lists, image_size=cfg.image_size, dataroot=cfg.dataroot)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default=cfg.data_dir)
    p.add_argument('--results_dir', default=cfg.results_dir)
    p.add_argument('--debug', action='store_true', help='200 steps with evaluation every 100 steps')
    args = p.parse_args()

    set_seed(cfg.seed)
    num_steps, eval_every = (200, 100) if args.debug else (cfg.num_steps, cfg.eval_every)

    trainer = Trainer(build_from_config(), dataset(args.data_dir, 'train'), dataset(args.data_dir, 'test'),
                      results_folder=args.results_dir, batch_size=cfg.batch_size,
                      gradient_accumulate_every=cfg.grad_accum, lr=cfg.lr, num_steps=num_steps,
                      ema_decay=cfg.ema_decay, eval_every=eval_every, save_last_every=cfg.save_last_every,
                      save_every=cfg.save_every, num_workers=cfg.num_workers)

    last = os.path.join(args.results_dir, 'last.pt')
    if os.path.exists(last):
        trainer.load(last)
    trainer.train()


if __name__ == '__main__':
    main()
