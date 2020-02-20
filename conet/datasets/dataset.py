import math
import os
import random
from os import path
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from skimage.color import gray2rgb
import cv2

import numpy as np
from torch.utils.data import Dataset
import pickle

from conet.config import get_cfg

train_aug = alb.Compose([
    # alb.RandomSizedCrop(min_max_height=(300, 500)),
    alb.RandomScale(),
    alb.HorizontalFlip(),
    alb.VerticalFlip(),
    alb.RandomBrightness(limit=0.01),
    alb.Rotate(limit=30),
    alb.PadIfNeeded(520, border_mode=cv2.BORDER_REFLECT101),
    alb.RandomCrop(512, 512),
    alb.Normalize(),
    # alb.pytorch.ToTensor(),
    ToTensorV2()
])

val_aug = alb.Compose([
    # alb.PadIfNeeded(512, border_mode=cv2.BORDER_REFLECT101),
    alb.Normalize(),
    alb.Resize(512, 512),
    ToTensorV2(),
])


class DukeOctDataset(Dataset):
    def __init__(self, split='train'):
        cfg = get_cfg()
        self.cfg = cfg
        self.data_dir = cfg.data_dir

        with open(path.join(cfg.data_dir, 'split.dp'), 'rb') as infile:
            self.d_split = pickle.load(infile)

        self.split = split
        self.d_basefp = self.d_split[split]

        if split == 'train':
            self.aug = train_aug
        elif split == 'val':
            self.aug = val_aug
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.d_basefp)

    def __getitem__(self, idx):
        carr = np.load(path.join(self.data_dir, self.d_basefp[idx]))
        img, label = carr[0], carr[1]

        img = gray2rgb(img)

        auged = self.aug(image=img, mask=label)

        auged['fname'] = self.d_basefp[idx]

        # img = auged['image']
        # print(img.shape)

        return auged
