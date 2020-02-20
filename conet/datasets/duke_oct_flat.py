import math
import os
import random
from os import path
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from skimage.color import gray2rgb
import cv2
from glob import glob

import imageio
import numpy as np
from torch.utils.data import Dataset
import pickle

from conet.config import get_cfg

train_aug = alb.Compose([
    # alb.RandomSizedCrop(min_max_height=(300, 500)),
    alb.RandomScale(),
    # alb.HorizontalFlip(),
    alb.VerticalFlip(),
    alb.RandomBrightness(limit=0.01),
    alb.Rotate(limit=30),
    # 224 548
    alb.PadIfNeeded(min_height=224, min_width=548, border_mode=cv2.BORDER_REFLECT101),
    alb.RandomCrop(224, 512),
    alb.Normalize(),
    # alb.pytorch.ToTensor(),
    ToTensorV2()
])

val_aug = alb.Compose([
    alb.PadIfNeeded(min_height=224, min_width=512, border_mode=cv2.BORDER_REFLECT101),
    alb.Normalize(),
    # alb.Resize(512, 512),
    alb.CenterCrop(224, 512),
    ToTensorV2(),
])

class DukeOctFlatDataset(Dataset):
    def __init__(self, split='train'):
        cfg = get_cfg()
        self.cfg = cfg
        self.data_dir = cfg.dme_flatten
        # self.data_dir = 

        # with open(path.join(cfg.data_dir, 'split.dp'), 'rb') as infile:
        #     self.d_split = pickle.load(infile)

        self.split = split

        # img_files = glob(path.join(self.data_dir, '*.jpg'))
        # img_bname = [path.basename(x).split('.')[0] for x in img_files]

        img_files = glob(path.join(self.data_dir, '*.npy'))
        img_bname = ['_'.join(path.basename(x).split('_')[:-1]) for x in img_files]

        subject_ids = [int(x.split('_')[1]) for x in img_bname]

        if split == 'train':
            self.bnames = [img_bname[i] for i in range(len(img_bname)) if subject_ids[i] < 6]
        else:
            self.bnames = [img_bname[i] for i in range(len(img_bname)) if subject_ids[i] >= 6]
        # self.d_basefp = self.d_split[split]


        self.imgs = []
        self.labels = []

        for b in self.bnames:
            self.imgs.append(imageio.imread(path.join(self.data_dir, f'{b}.jpg')))
            self.labels.append(np.load(path.join(self.data_dir, f'{b}_label.npy')))

        if split == 'train':
            self.aug = train_aug
        elif split == 'val':
            self.aug = val_aug
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.bnames)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        img = gray2rgb(img)

        auged = self.aug(image=img, mask=label)

        # auged['fname'] = self.d_basefp[idx]
        label = auged['mask']
        loss_mask = (label !=-1).float()
        # loss_mask = torch.from_numpy(loss_mask)
        auged['fname'] = self.bnames[idx]

        auged['loss_mask'] = loss_mask
        # img = auged['image']
        # print(img.shape)
        
        return auged