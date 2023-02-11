import random
import os
from collections import OrderedDict
from functools import partial
import pickle
from pprint import pformat
import os.path as osp

import torch
import numpy as np

from .utils import check_isfile


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_snap(model, optimizer, scheduler, epoch, log_path):
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'epoch': epoch}

    snap_name = f'{log_path}/snap_{epoch}.pth'
    print(f'==> saving checkpoint to {snap_name}')
    torch.save(checkpoint, snap_name)


@torch.no_grad()
def compute_accuracy(pred_cats, gt_cats, reduce_mean=True, **kwargs):
    pred_cats = torch.argmax(pred_cats, dim=1)
    if reduce_mean:
        return torch.mean((pred_cats == gt_cats).float()).item()

    return torch.sum((pred_cats == gt_cats).int()).item()


def put_on_device(items, device):
    for i, item in enumerate(items):
        items[i] = item.to(device)
    return items


def load_checkpoint(fpath):
    r"""Loads checkpoint. Imported from openvinotoolkit/deep-object-reid.
    Args:
        fpath (str): path to checkpoint.
    Returns:
        dict
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers):
    if discarded_layers:
        print(
            '** The following layers are discarded '
            'due to unmatched keys or layer size: {}'.
            format(pformat(discarded_layers))
        )
    if unmatched_layers:
        print(
            '** The following layers were not loaded from checkpoint: {}'.
            format(pformat(unmatched_layers))
        )


def load_pretrained_weights(model, file_path='', pretrained_dict=None, extra_prefix=''):
    r"""Loads pretrianed weights to model. Imported from openvinotoolkit/deep-object-reid.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        file_path (str): path to pretrained weights.
    """
    def _remove_prefix(key, prefix):
        prefix = prefix + '.'
        if key.startswith(prefix):
            key = key[len(prefix):]
        return key

    if file_path:
        check_isfile(file_path)
    checkpoint = (load_checkpoint(file_path)
                       if not pretrained_dict
                       else pretrained_dict)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        k = extra_prefix + _remove_prefix(k, 'module')

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    message = file_path if file_path else "pretrained dict"
    unmatched_layers = sorted(set(model_dict.keys()) - set(new_state_dict))
    if len(matched_layers) == 0:
        print(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually'.format(message)
        )
        _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)

        raise RuntimeError(f'The pretrained weights {message} cannot be loaded')
    print(
        'Successfully loaded pretrained weights from "{}"'.
        format(message)
    )
    _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)


def resume_from(model, chkpt_path, optimizer=None, scheduler=None):
    print(f'Loading checkpoint from "{chkpt_path}"')
    checkpoint = load_checkpoint(chkpt_path)
    if 'state_dict' in checkpoint:
        load_pretrained_weights(model, pretrained_dict=checkpoint['state_dict'])
    else:
        load_pretrained_weights(model, pretrained_dict=checkpoint)
    print('Loaded model weights')
    if optimizer is not None and 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded optimizer')
    if scheduler is not None and 'scheduler' in checkpoint.keys():
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('Loaded scheduler')
    if 'epoch' in checkpoint:
        # since saved in the end of an epoch we increment it by 1
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
        print("Warning. The epoch has not been restored.")
    print('Last epoch = {}'.format(start_epoch))
    if 'rank1' in checkpoint.keys():
        print('Last rank1 = {:.1%}'.format(checkpoint['rank1']))
    return start_epoch