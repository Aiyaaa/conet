import math
import os
import random
from os import path
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from skimage.color import gray2rgb
import skimage
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle

from conet.config import get_cfg

train_size_aug = alb.Compose([
    # alb.RandomSizedCrop(min_max_height=(300, 500)),
    
    alb.HorizontalFlip(),
    alb.VerticalFlip(),
    # alb.RandomBrightness(limit=0.01),
    alb.RandomScale(),
    alb.ElasticTransform(),
    alb.Rotate(limit=50),
    
    alb.PadIfNeeded(530, border_mode=cv2.BORDER_REFLECT101),
    
    alb.RandomCrop(512, 512),
    # alb.Normalize(),
    # alb.pytorch.ToTensor(),
    # ToTensorV2()
])
train_content_aug = alb.Compose([
    alb.MedianBlur(),
    alb.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
    
    alb.RandomBrightnessContrast(brightness_limit=0.1),
    alb.Normalize(),
    # ToTensorV2()
])

val_aug = alb.Compose([
    # alb.PadIfNeeded(512, border_mode=cv2.BORDER_REFLECT101),
    # alb.Normalize(),
    alb.Resize(512, 512),
    # ToTensorV2(),
])
val_c_aug = alb.Compose([
    alb.Normalize(),
    # ToTensorV2()
])

class DukeOctSPDataset(Dataset):
    def __init__(self, split='train'):
        cfg = get_cfg()
        self.cfg = cfg
        self.data_dir = cfg.data_sp_dir

        with open(path.join(cfg.data_dir, 'split.dp'), 'rb') as infile:
            self.d_split = pickle.load(infile)

        self.split = split
        self.d_basefp = self.d_split[split]

        if split == 'train':
            self.b_aug = train_size_aug
            self.c_aug = train_content_aug
        elif split == 'val':
            self.b_aug = val_aug
            self.c_aug = val_c_aug
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.d_basefp)

    def __getitem__(self, idx):
        carr = np.load(path.join(self.data_dir, self.d_basefp[idx]))
        img, label = carr[0:-1], carr[-1]

        # c x y => xyc
        img = np.transpose(img, (1, 2, 0))

        # img = gray2rgb(img)

        auged = self.b_aug(image=img, mask=label)

        img = auged['image']
        label = auged['mask']

        soft_label = img[:, :, 1:]
        image = img[:, :, 0]

        # print(image.shape, image.max(), image.min())

        image = np.clip(image, 0, 255).astype('uint8')

        # image = skimage.img_as_ubyte(image)

        image = gray2rgb(image)
        image = self.c_aug(image=image)['image'] # normi

        # image = alb.Normalize()(image)['image']

        image = np.transpose(image, (2, 0, 1))
        soft_label = np.transpose(soft_label, (2, 0, 1))

        image = torch.from_numpy(image)
        soft_label = torch.from_numpy(soft_label).float()
        label = torch.from_numpy(label)

        # img = auged['image']
        # print(img.shape)
        
        return {
            'image': image,
            'softlabel': soft_label,
            'mask': label,
            'fname': self.d_basefp[idx]
        }