import argparse
import sys
import time
import os.path as osp
from shutil import copyfile

from simplecls.utils import read_py_config, Logger, check_isfile
from simplecls.torch_utils import set_random_seed, resume_from
from simplecls.builders import build_model, build_loader
from simplecls.quantizer import Quantizer
from simplecls.evaluator import Evaluator


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
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda','cpu', 'xpu'],
                        help='choose device to train on')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32','fp16'],
                        help='choose training precision (works for non-cpu devices)')
    args = parser.parse_args()
    cfg = read_py_config(args.config)
    reset_config(cfg, args)
    # translate output to log file
    log_name = 'quantize.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.output_dir, log_name))

    copyfile(args.config, osp.join(cfg.output_dir, 'dumped_config.py'))

    set_random_seed(cfg.utils.random_seeds)
    net = build_model(cfg)
    net.to(args.device)

    _, val_loader = build_loader(cfg)

    quantizer = Quantizer(model=net,
                          val_loader=val_loader,
                          cfg=cfg,
                          max_batches=1,  # Set a limit for quantization batches
                          device=args.device,
                          half_precision=cfg.half_precision)

    q_model = quantizer.run()

    evaluator = Evaluator(model=q_model,
                          val_loader=val_loader,
                          cfg=cfg,
                          device=args.device,
                          max_epoch=cfg.data.max_epochs,
                          half_precision=cfg.half_precision)
    evaluator.run()


if __name__ == '__main__':
    main()