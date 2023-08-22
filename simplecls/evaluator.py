from functools import partial
import time
import torch
from dataclasses import dataclass
from tqdm import tqdm

from .utils import AverageMeter
from .torch_utils import compute_accuracy, put_on_device


@dataclass
class Evaluator:
    model: object
    val_loader: object
    cfg: dict
    max_epoch: int
    writer: object = None
    device: str = 'cuda'
    debug: bool = False
    debug_steps: int = 30
    half_precision: bool = False

    @torch.no_grad()
    def val(self, epoch=None):
        ''' procedure launching main validation '''
        acc_meter = AverageMeter()
        cuda_half_p = self.half_precision and torch.cuda.is_available()
        if cuda_half_p:
            autocaster = partial(torch.cuda.amp.autocast, enabled=True)
        else:
            autocaster = partial(torch.cuda.amp.autocast, enabled=False)

        xpu_half_p = self.half_precision and hasattr(torch, 'xpu')
        if xpu_half_p:
            autocaster = partial(torch.xpu.amp.autocast, enabled=True, dtype=torch.bfloat16)

        # switch to eval mode
        self.model.eval()
        loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=False)
        start = time.time()
        total_compute_time = 0.
        for it, (imgs, gt_cats) in loop:
            # put image and keypoints on the appropriate device
            compute_start = time.time()
            imgs, gt_cats = put_on_device([imgs, gt_cats], self.device)
            # compute output and loss
            with autocaster():
                pred_cats = self.model(imgs)
            top1 = compute_accuracy(pred_cats, gt_cats, reduce_mean=False)
            acc_meter.update(top1, pred_cats.shape[0])
            total_compute_time += time.time() - compute_start

            if epoch is not None:
                # update progress bar
                loop.set_description(f'Val Epoch [{epoch}/{self.max_epoch}]')
                loop.set_postfix(acc_avg=acc_meter.avg)

            if self.debug and it == self.debug_steps:
                break

        if epoch is not None and self.writer is not None:
            # write to writer for tensorboard
            self.writer.add_scalar('Val/ACC', acc_meter.avg, global_step=epoch)

        print(f'Top-1 accuracy: {acc_meter.avg}')
        print(f'Val time: {time.time() - start}')
        print(f'Val compute time: {total_compute_time}')

        return acc_meter.avg
