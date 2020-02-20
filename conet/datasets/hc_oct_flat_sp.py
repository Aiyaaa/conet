import math
import os
import pickle
import random
from glob import glob
from os import path

import albumentations as alb
import cv2
import numpy as np
import skimage
import torch
import imageio
from albumentations.pytorch import ToTensorV2
from skimage.color import gray2rgb
from torch.utils.data import Dataset

from conet.config import get_cfg

train_size_aug = alb.Compose([
    # alb.RandomSizedCrop(min_max_height=(300, 500)),
    alb.RandomScale(),
    # alb.HorizontalFlip(),
    alb.VerticalFlip(),
    # alb.RandomBrightness(limit=0.01),
    
    alb.ElasticTransform(),
    alb.Rotate(limit=30),
    
    alb.PadIfNeeded(min_height=128+10, min_width=1024+100, border_mode=cv2.BORDER_REFLECT101),
    alb.RandomCrop(128, 1024),
    # alb.Normalize(),
    # alb.pytorch.ToTensor(),
    # ToTensorV2()
])
train_content_aug = alb.Compose([
    # alb.MedianBlur(3),
    alb.GaussianBlur(3),
    alb.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5),
    
    alb.RandomBrightnessContrast(brightness_limit=0.01),
    alb.Normalize(),
    # ToTensorV2()
])

val_aug = alb.Compose([
    # alb.PadIfNeeded(512, border_mode=cv2.BORDER_REFLECT101),
    # alb.Normalize(),
    # alb.Resize(512, 512),
    # alb.PadIfNeeded(min_height=224, min_width=512, border_mode=cv2.BORDER_REFLECT101),
    # alb.CenterCrop(224, 512),
    alb.PadIfNeeded(min_height=128, min_width=1024, border_mode=cv2.BORDER_REFLECT101),
    alb.CenterCrop(128, 1024),
    # ToTensorV2(),
])
val_c_aug = alb.Compose([
    alb.Normalize(),
    
    # ToTensorV2()
])

train_aug_f = alb.Compose([
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

val_aug_f = alb.Compose([
    alb.PadIfNeeded(min_height=224, min_width=512, border_mode=cv2.BORDER_REFLECT101),
    alb.Normalize(),
    # alb.Resize(512, 512),
    alb.CenterCrop(224, 512),
    ToTensorV2(),
])

class HcOctFlatSPDataset(Dataset):
    def __init__(self, split='train'):
        cfg = get_cfg()
        self.cfg = cfg
        # self.data_dir = cfg.dme_flatten_sp
        if split == 'train':
            self.data_dir = cfg.hc_train_sp
        else:
            self.data_dir = cfg.hc_test_sp

        # with open(path.join(cfg.data_dir, 'split.dp'), 'rb') as infile:
        #     self.d_split = pickle.load(infile)

        self.split = split
        

        img_files = glob(path.join(self.data_dir, '*_label.npy'))
        img_bname = ['_'.join(path.basename(x).split('_')[:-1]) for x in img_files]
        self.bnames = img_bname
       

        if split == 'train':
            self.b_aug = train_size_aug
            self.c_aug = train_content_aug
        elif split == 'val':
            self.b_aug = val_aug
            self.c_aug = val_c_aug
        else:
            raise NotImplementedError
        self.cache = {}
        # for idx in range(len(self)):
        #     bname = self.bnames[idx]
        #     img_fp = path.join(self.data_dir, f'{bname}.jpg')
        #     label_fp = path.join(self.data_dir, f'{bname}_label.npy')
        #     softlabel_fp = path.join(self.data_dir, f'{bname}_softlabel.npy')

        #     img = imageio.imread(img_fp)
        #     label = np.load(label_fp)
        #     softlabel = np.load(softlabel_fp)

        #     self.cache.append((img_fp, img, label, softlabel))

    def __len__(self):
        # return len(self.d_basefp)
        return len(self.bnames)

    def __getitem__(self, idx):
        # carr = np.load(path.join(self.data_dir, self.d_basefp[idx]))
        # carr = np.load(self.bnames[idx])
        if idx in self.cache.keys():
            img_fp, img, label, softlabel = self.cache[idx]
        else:
            bname = self.bnames[idx]
            img_fp = path.join(self.data_dir, f'{bname}.jpg')
            label_fp = path.join(self.data_dir, f'{bname}_label.npy')
            softlabel_fp = path.join(self.data_dir, f'{bname}_softlabel.npy')

            img = imageio.imread(img_fp)
            label = np.load(label_fp)
            softlabel = np.load(softlabel_fp)

            self.cache[idx] = (img_fp, img, label, softlabel)

        # img_fp, img, label, softlabel = self.cache[idx]

        # img = gray2rgb(img)
        # if self.split == 'train':
        #     auged = train_aug_f(image=img, mask=label)
        # else:
        #     auged = val_aug_f(image=img, mask=label)
        # auged['fname'] = img_fp
        # auged['softlabel'] = torch.tensor(0.)
        # return auged
        

        # img = np.transpose(img, (1, 2, 0))
        # to channel last
        softlabel = np.transpose(softlabel, (1, 2, 0))
        img = np.expand_dims(img, axis=-1)

        img_a = np.concatenate([img, softlabel], axis=-1)

        # img = gray2rgb(img)

        auged = self.b_aug(image=img_a, mask=label)

        img = auged['image']
        label = auged['mask']

        softlabel = img[:, :, 1:]
        image = img[:, :, 0]

        # print(image.shape, image.max(), image.min())

        image = np.clip(image, 0, 255).astype('uint8')

        # image = skimage.img_as_ubyte(image)

        image = gray2rgb(image)
        image = self.c_aug(image=image)['image'] # normi

        # image = alb.Normalize()(image)['image']

        image = np.transpose(image, (2, 0, 1))
        softlabel = np.transpose(softlabel, (2, 0, 1))

        

        image = torch.from_numpy(image)
        softlabel = torch.from_numpy(softlabel).float()
        label = torch.from_numpy(label)


        # img = auged['image']
        # print(img.shape)
        
        return {
            'image': image,
            'softlabel': softlabel,
            'mask': label,
            'fname': img_fp,
            
        }
