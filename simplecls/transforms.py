import math
import random
import cv2 as cv
import numpy as np
import torch
from albumentations.core.transforms_interface import BasicTransform, ImageOnlyTransform, DualTransform, to_tuple


class ConvertColor(ImageOnlyTransform):
    """Converting color of the image
    """
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        if img.shape[0] == 1:
            return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)


class RandomRescale(ImageOnlyTransform):
    """Rescaling image
    """
    def __init__(self, scale_limit=0.1, interpolation=cv.INTER_LINEAR, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale_limit = to_tuple(scale_limit, bias=0)
        self.interpolation = interpolation

    def get_params(self):
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale=0, interpolation=cv.INTER_LINEAR, **params):
        h, w = img.shape[:2]
        rot_mat = cv.getRotationMatrix2D((w*0.5, h*0.5), 0, scale)
        image = cv.warpAffine(img, rot_mat, (w, h), flags=interpolation)
        return image

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "scale_limit": to_tuple(self.scale_limit, bias=-1.0)}


class RandomRotate(ImageOnlyTransform):
    """Rotate image
    """
    def __init__(self, angle_limit=0.1, interpolation=cv.INTER_LINEAR, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.angle_limit = to_tuple(angle_limit)
        self.interpolation = interpolation

    def get_params(self):
        return {"angle": random.uniform(self.angle_limit[0], self.angle_limit[1])}

    def apply(self, img, angle=0, interpolation=cv.INTER_LINEAR, **params):
        h, w = img.shape[:2]
        scale = self._get_scale_by_angle(angle, h, w)
        rot_mat = cv.getRotationMatrix2D((w*0.5, h*0.5), angle, scale)
        image = cv.warpAffine(img, rot_mat, (w, h), flags=interpolation)
        return image

    @staticmethod
    def _get_scale_by_angle(angle, h, w):
        rad_angle = math.radians(angle)
        cos = math.cos(rad_angle) - 1
        sin = math.sin(rad_angle)
        delta_h = w / 2 * cos + h / 2 * sin
        delta_w = w / 2 * sin + h / 2 * cos
        return max(w / (w + 2 * abs(delta_w)), h / (h + 2 * abs(delta_h)))

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "scale_limit": to_tuple(self.angle_limit)}


class ToTensor(BasicTransform):
    """Converting iamge to tensor
    """
    def __init__(self, img_shape, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.img_final_shape = img_shape

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1)).float()