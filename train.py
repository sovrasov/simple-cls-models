import argparse
import sys
import time
import os.path as osp
from shutil import copyfile

from simplecls.utils import read_py_config, Logger, check_isfile
from simplecls.torch_utils import set_random_seed, resume_from
from simplecls.builders import build_model, build_optimizer, build_scheduler, build_loss, build_loader
from simplecls.trainer import Trainer
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
    net.to(args.device)

    optimizer = build_optimizer(cfg, net)
    scheduler = build_scheduler(cfg, optimizer)

    if args.device == 'xpu':
        dtype = torch.bfloat16 if cfg.half_precision else torch.float32
        net, optimizer = ipex.optimize(net, optimizer=optimizer, dtype=dtype)

    if cfg.model.resume:
        if check_isfile(cfg.model.resume):
            start_epoch = resume_from(net, cfg.model.resume, optimizer=optimizer, scheduler=scheduler)
        else:
            raise RuntimeError("the checkpoint isn't found ot can't be loaded!")
    else:
        start_epoch = 0

    criterion = build_loss(cfg)
    train_loader, val_loader = build_loader(cfg)
    train_step = (start_epoch - 1)*len(train_loader) if start_epoch > 1 else 0

    trainer = Trainer(model=net,
                      train_loader=train_loader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      loss=criterion,
                      max_epoch=cfg.data.max_epochs,
                      log_path=cfg.output_dir,
                      device=args.device,
                      save_freq=cfg.utils.save_freq,
                      print_freq=cfg.utils.print_freq,
                      train_step=train_step,
                      half_precision=cfg.half_precision)

    evaluator = Evaluator(model=net,
                          val_loader=val_loader,
                          cfg=cfg,
                          device=args.device,
                          max_epoch=cfg.data.max_epochs,
                          half_precision=cfg.half_precision)
    # main loop
    if cfg.regime.type == "evaluation":
        evaluator.run()
    else:
        assert cfg.regime.type == "training"
        if cfg.model.resume:
            evaluator.val()
        for epoch in range(start_epoch, cfg.data.max_epochs):
            is_last_epoch = epoch == cfg.data.max_epochs - 1
            trainer.train(epoch, is_last_epoch)
            if epoch % cfg.utils.eval_freq == 0 or is_last_epoch:
                evaluator.val(epoch)


if __name__ == '__main__':
    main()