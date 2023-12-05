import argparse
import sys
import time
import os.path as osp
from shutil import copyfile

from pytorch_lightning import Trainer
from simplecls.lt_module import LightningModelWrapper

from simplecls.utils import read_py_config, Logger, check_isfile
from simplecls.torch_utils import set_random_seed, resume_from
from simplecls.builders import build_model, build_optimizer, build_scheduler, build_loss, build_loader
from simplecls.evaluator import Evaluator

try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except:
    has_ipex = False
import torch


def reset_config(cfg, args):
    if args.root:
        cfg['data']['root'] = args.root
    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.precision == 'fp16':
        cfg['half_precision'] = True
    else:
        cfg['half_precision'] = False


def main():
    parser = argparse.ArgumentParser(description='PyTorch cls training')
    parser.add_argument('--root', type=str, default='', help='path to root folder')
    parser.add_argument('--output_dir', type=str, default='', help='directory to store training artifacts')
    parser.add_argument('--config', type=str, default='./configs/default_config.py', help='path to config')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu', 'xpu'],
                        help='choose device to train on')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32','fp16'],
                        help='choose training precision (works for non-cpu devices)')
    args = parser.parse_args()
    cfg = read_py_config(args.config)
    reset_config(cfg, args)
    # translate output to log file
    log_name = 'train.log' if cfg.regime.type == 'training' else 'test.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.output_dir, log_name))

    copyfile(args.config, osp.join(cfg.output_dir, 'dumped_config.py'))

    set_random_seed(cfg.utils.random_seeds)


    net = build_model(cfg)
    optimizer = build_optimizer(cfg, net)
    scheduler = build_scheduler(cfg, optimizer)
    criterion = build_loss(cfg)
    train_loader, val_loader = build_loader(cfg)

    model = LightningModelWrapper(net, optimizer, criterion, scheduler)
    trainer = Trainer(max_epochs=cfg.data.max_epochs, check_val_every_n_epoch=cfg.utils.eval_freq,
                      accelerator="xpu", devices=1, strategy="xpu_single")


    evaluator = Evaluator(model=net,
                          val_loader=val_loader,
                          cfg=cfg,
                          device=args.device,
                          max_epoch=cfg.data.max_epochs,
                          half_precision=cfg.half_precision)
    # main loop
    if cfg.regime.type == "evaluation":
        evaluator.val()
    else:
        assert cfg.regime.type == "training"
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()