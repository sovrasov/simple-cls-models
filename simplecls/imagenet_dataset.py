import os
from os import path as osp

import cv2 as cv
from torch.utils.data import Dataset
import torch

class ClassificationImageFolder(Dataset):
    """Classification dataset representing raw folders without annotation files.
    """

    def __init__(self, root, transform=None, filter_classes=None):
        super().__init__()
        self.root = root
        assert osp.isdir(root)
        self.data, self.classes = self.load_annotation(self.root, filter_classes)
        self.num_classes = len(self.classes)
        self.transform = transform

    @staticmethod
    def load_annotation(data_dir, filter_classes=None):
        ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.gif')
        def is_valid(filename):
            return not filename.startswith('.') and filename.lower().endswith(ALLOWED_EXTS)

        def find_classes(folder, filter_names=None):
            if filter_names:
                classes = [d.name for d in os.scandir(folder) if d.is_dir() and d.name in filter_names]
            else:
                classes = [d.name for d in os.scandir(folder) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return class_to_idx

        class_to_idx = find_classes(data_dir, filter_classes)

        out_data = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = osp.join(data_dir, target_class)
            if not osp.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = osp.join(root, fname)
                    if is_valid(path):
                        out_data.append((path, class_index))

        if not out_data:
            print('Failed to locate images in folder ' + data_dir + f' with extensions {ALLOWED_EXTS}')

        return out_data, class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        img_path, category = self.data[indx]
        image = cv.imread(img_path)

        if self.transform:
            transformed = self.transform(image=image)
            transformed_image = transformed['image']
            assert isinstance(transformed_image, torch.Tensor)
        else:
            transformed_image = image

        return transformed_image, category
