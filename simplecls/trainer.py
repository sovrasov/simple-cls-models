import time
import datetime

from tqdm import tqdm
from dataclasses import dataclass

from .utils import AverageMeter
from .torch_utils import compute_accuracy, put_on_device, save_snap


@dataclass(init=True)
class Trainer:
    model: object
    train_loader: object
    optimizer: object
    scheduler: object
    loss: object
    max_epoch : int
    log_path : str
    writer: object = None
    device : str ='cuda'
    save_chkpt: bool = True
    debug: bool = False
    debug_steps: int = 30
    save_freq: int = 10
    print_freq: int = 10
    train_step: int = 0

    def train(self, epoch, is_last_epoch):
        ''' procedure launching main training'''

        losses = AverageMeter()
        acc_meter = AverageMeter()
        batch_time = AverageMeter()
        compute_time = AverageMeter()

        # switch to train mode and train one epoch
        self.model.train()
        self.num_iters = len(self.train_loader)
        start = time.time()
        iter_start = time.time()
        total_compute_time = 0.
        loop = tqdm(enumerate(self.train_loader), total=self.num_iters, leave=False)
        for it, (imgs, gt_cats) in loop:
            # put image and keypoints on the appropriate device
            compute_start = time.time()
            imgs, gt_cats = put_on_device([imgs, gt_cats], self.device)
            # compute output and loss
            pred_cats = self.model(imgs)
            # get parsed loss
            loss = self.loss(pred_cats, gt_cats)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # measure metrics
            acc = compute_accuracy(pred_cats, gt_cats)
            acc_meter.update(acc)
            # record loss
            losses.update(loss.item(), imgs.size(0))
            # write to writer for tensorboard
            if self.writer is not None:
                self.writer.add_scalar('Train/loss', loss.item(), global_step=self.train_step)
                self.writer.add_scalar('Train/avg_acc', acc_meter.avg, global_step=self.train_step)
            self.train_step += 1
            # update progress bar
            loop.set_description(f'Epoch [{epoch}/{self.max_epoch}]')
            loop.set_postfix(loss=loss.item(),
                             avr_loss = losses.avg,
                             acc=acc,
                             acc_avg = acc_meter.avg,
                             lr=self.optimizer.param_groups[0]['lr'])
            # compute eta
            compute_time.update(time.time() - compute_start)
            total_compute_time += time.time() - compute_start
            batch_time.update(time.time() - iter_start)
            nb_this_epoch = self.num_iters - (it + 1)
            nb_future_epochs = (self.max_epoch - (epoch + 1)) * self.num_iters
            eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            if ((it % self.print_freq == 0) or (it == self.num_iters-1)):
                print(
                        'epoch: [{0}/{1}][{2}/{3}]\t'
                        'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'eta {eta}\t'
                        'cls acc {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                        'loss {losses.avg:.5f}\t'
                        'lr {lr:.6f}'.format(
                            epoch,
                            self.max_epoch,
                            it,
                            self.num_iters,
                            batch_time=batch_time,
                            eta=eta_str,
                            accuracy=acc_meter,
                            losses=losses,
                            lr=self.optimizer.param_groups[0]['lr'])
                        )

            iter_start = time.time()
            if (self.debug and it == self.debug_steps):
                break

        if self.save_chkpt and (epoch % self.save_freq == 0 or is_last_epoch) and not self.debug:
            save_snap(self.model, self.optimizer, self.scheduler, epoch, self.log_path)
        # do scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        print(f'Final avg batch time: {batch_time.avg}')
        print(f'Final avg batch compute time: {compute_time.avg}')
        print(f'Epoch time: {time.time() - start}')
        print(f'Epoch time compute: {total_compute_time}')