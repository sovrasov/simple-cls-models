import random
from pathlib import Path

from torch.utils.data import DataLoader
import albumentations as A
import numpy as np

from ..imagenet_dataset import ClassificationImageFolder
from ..transforms import ConvertColor, ToTensor, RandomRescale, RandomRotate


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(random.getstate()[1][0] + worker_id + 1)


def build_loader(config):

    train_transform, test_transform = build_augmentations(cfg=config)

    train_dataset = ClassificationImageFolder(Path(config.data.root) / 'train', train_transform)
    val_dataset = ClassificationImageFolder(Path(config.data.root) / 'val', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.data.train_batch_size,
                              shuffle=True, num_workers=config.data.num_workers,
                              worker_init_fn=worker_init_fn)

    val_loader = DataLoader(val_dataset, batch_size=config.data.val_batch_size, shuffle=False,
                            num_workers=config.data.num_workers,
                            worker_init_fn=worker_init_fn)

    return train_loader, val_loader


TRANSFORMS_REGISTRY = {
        'convert_color': ConvertColor,
        'random_rescale': RandomRescale,
        'resize': A.Resize,
        'horizontal_flip': A.HorizontalFlip,
        'hue_saturation_value': A.HueSaturationValue,
        'rgb_shift': A.RGBShift,
        'random_brightness_contrast': A.RandomBrightnessContrast,
        'color_jitter': A.ColorJitter,
        'blur': A.Blur,
        'normalize': A.augmentations.transforms.Normalize,
        'to_tensor': ToTensor,
        'one_of': A.OneOf,
        'random_rotate': RandomRotate,
    }


def build_transforms_list(transforms_config):
    transforms = []
    for t, args in transforms_config:
        if t == 'one_of':
            transforms.append(TRANSFORMS_REGISTRY[t](build_transforms_list(args.transforms), p=args.p))
        else:
            transforms.append(TRANSFORMS_REGISTRY[t](**args))
    return transforms

def build_augmentations(cfg):
    train_transform = A.Compose(build_transforms_list(cfg.train_data_pipeline))
    test_transform = A.Compose(build_transforms_list(cfg.test_data_pipeline))
    return train_transform, test_transform