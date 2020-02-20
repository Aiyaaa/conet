import multiprocessing as mp
# mp.set_start_method('spawn')

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


# https://github.com/albumentations-team/albumentations/pull/511
# Fix grid distortion bug. #511
# GridDistortion bug修复.....


train_size_aug = alb.Compose([
    # alb.RandomSizedCrop(min_max_height=(300, 500)),
    alb.PadIfNeeded(min_height=100, min_width=600, border_mode=cv2.BORDER_REFLECT101),
    alb.Rotate(limit=6),
    alb.RandomScale(scale_limit=0.05,),

    alb.ElasticTransform(),
    # alb.GridDistortion(p=1, num_steps=20, distort_limit=0.5),
    # alb.GridDistortion(num_steps=10, p=1),

    # alb.OneOf([
    #     alb.OpticalDistortion(),
        
    # ]),
    # alb.MaskDropout(image_fill_value=0, mask_fill_value=-1,p=0.3),
    alb.HorizontalFlip(),
    # alb.VerticalFlip(),
    
    # alb.RandomBrightness(limit=0.01),
    
    
    
    
    alb.PadIfNeeded(min_height=224, min_width=512, border_mode=cv2.BORDER_REFLECT101),
    alb.RandomCrop(224, 512),
    # alb.Normalize(),
    # alb.pytorch.ToTensor(),
    # ToTensorV2()
])
train_content_aug = alb.Compose([
    # alb.MedianBlur(3),
    # alb.GaussianBlur(3),
    alb.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5),
    
    alb.RandomBrightnessContrast(brightness_limit=0.05),
    alb.Normalize(),
    # ToTensorV2()
])

val_aug = alb.Compose([
    # alb.PadIfNeeded(512, border_mode=cv2.BORDER_REFLECT101),
    # alb.Normalize(),
    # alb.Resize(512, 512),
    alb.PadIfNeeded(min_height=224, min_width=512, border_mode=cv2.BORDER_REFLECT101),
    alb.CenterCrop(224, 512),
    # ToTensorV2(),
])
val_c_aug = alb.Compose([
    alb.Normalize(),
    
    # ToTensorV2()
])

# train_aug_f = alb.Compose([
#     # alb.RandomSizedCrop(min_max_height=(300, 500)),
#     alb.RandomScale(),
#     # alb.HorizontalFlip(),
#     alb.VerticalFlip(),
#     alb.RandomBrightness(limit=0.01),
#     alb.Rotate(limit=30),
#     # 224 548
#     alb.PadIfNeeded(min_height=224, min_width=548, border_mode=cv2.BORDER_REFLECT101),
#     alb.RandomCrop(224, 512),
#     alb.Normalize(),
#     # alb.pytorch.ToTensor(),
#     ToTensorV2()
# ])

# val_aug_f = alb.Compose([
#     alb.PadIfNeeded(min_height=224, min_width=512, border_mode=cv2.BORDER_REFLECT101),
#     alb.Normalize(),
#     # alb.Resize(512, 512),
#     alb.CenterCrop(224, 512),
#     ToTensorV2(),
# ])

class DukeOctFlatSPDataset(Dataset):
    def __init__(self, split='train', n_seg=0):
        cfg = get_cfg()
        self.cfg = cfg
        self.data_dir = path.join(cfg.dme_flatten_sp, str(n_seg))

        print(f'Load data from {self.data_dir}')
        

        # with open(path.join(cfg.data_dir, 'split.dp'), 'rb') as infile:
        #     self.d_split = pickle.load(infile)

        self.split = split
        

        data_files = glob(path.join(self.data_dir, '*.jpg'))
        # img_bname = ['_'.join(path.basename(x).split('_')[:-1]) for x in img_files]

        data_bnames = [path.basename(x).split('.')[0] for x in data_files]
        # self.data_bnames = data_bnames
        subject_ids = [int(x.split('_')[1]) for x in data_bnames]

        if split == 'train':
            self.bnames = [data_bnames[i] for i in range(len(data_files)) if subject_ids[i] < 6]
        else:
            self.bnames = [data_bnames[i] for i in range(len(data_files)) if subject_ids[i] >= 6]

        if split == 'train':
            self.b_aug = train_size_aug
            self.c_aug = train_content_aug
        elif split == 'val':
            self.b_aug = val_aug
            self.c_aug = val_c_aug
        else:
            raise NotImplementedError
        self.cache = []
        for idx in range(len(self)):
            bname = self.bnames[idx]
            img_fp = path.join(self.data_dir, f'{bname}.jpg')
            label_fp = path.join(self.data_dir, f'{bname}_label.npy')
            softlabel_fp = path.join(self.data_dir, f'{bname}_softlabel.npy')

            img = imageio.imread(img_fp)
            label = np.load(label_fp)
            softlabel = np.load(softlabel_fp)

            self.cache.append((img_fp, img, label, softlabel))

    def __len__(self):
        # return len(self.d_basefp)
        return len(self.bnames)

    def __getitem__(self, idx):
        # carr = np.load(path.join(self.data_dir, self.d_basefp[idx]))
        # carr = np.load(self.bnames[idx])
        # if idx in self.cache.keys():
        #     img_fp, img, label, soft_label = self.cache[idx]
        # else:
        #     bname = self.bnames[idx]
        #     img_fp = path.join(self.data_dir, f'{bname}.jpg')
        #     label_fp = path.join(self.data_dir, f'{bname}_label.npy')
        #     softlabel_fp = path.join(self.data_dir, f'{bname}_softlabel.npy')

        #     img = imageio.imread(img_fp)
        #     label = np.load(label_fp)
        #     softlabel = np.load(softlabel_fp)

        #     self.cache[idx] = (img_fp, img, label, softlabel)

        img_fp, img, label, softlabel = self.cache[idx]

        img_fp, img, label, softlabel = img_fp, img.copy(), label.copy(), softlabel.copy()

        # img = gray2rgb(img)
        # if self.split == 'train':
        #     auged = train_aug_f(image=img, mask=label)
        # else:
        #     auged = val_aug_f(image=img, mask=label)
        # auged['fname'] = img_fp
        # auged['softlabel'] = torch.tensor(0.)
        # return auged
        

        # img = np.transpose(img, (1, 2, 0))
        softlabel = np.transpose(softlabel, (1, 2, 0))
        img = np.expand_dims(img, axis=-1)

        img_a = np.concatenate([img, softlabel], axis=-1)

        # img = gray2rgb(img)

        # grid_distortion 可能不支持负数
        
        label[label == -1] = 255

        auged = self.b_aug(image=img_a, mask=label)

        img = auged['image']
        label = auged['mask']

        label[label == 255] = -1

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

        loss_mask = (label !=-1).astype("float")

        image = torch.from_numpy(image)
        softlabel = torch.from_numpy(softlabel).float()
        label = torch.from_numpy(label)

        loss_mask = torch.from_numpy(loss_mask)

        # img = auged['image']
        # print(img.shape)
        
        return {
            'image': image,
            'softlabel': softlabel,
            'mask': label,
            'fname': img_fp,
            'loss_mask': loss_mask
        }



if __name__ == "__main__":
    from skimage import segmentation, color, filters, exposure
    import skimage
    import os 
    from os import path
    import imageio
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    import random 
    
    np.random.seed(42)
    random.seed(42)
    save_dir = '/data1/hangli/oct/debug'
    os.makedirs(save_dir, exist_ok=True)

    cmap = plt.cm.get_cmap('jet')
    n_seg = 1200
    training_dataset = DukeOctFlatSPDataset(split='train', n_seg=n_seg)
    # val_dataset = DukeOctFlatSPDataset(split='val', n_seg=n_seg)
    
    data_loader = DataLoader(training_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    for t in range(40):
        for bidx, batch in enumerate(data_loader):
            data = batch['image']
            target = batch['mask']
            for b_i in range(len(data)):
                img = data[b_i]
                img = img.permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())
                img = skimage.img_as_ubyte(img)
                mask = target[b_i]
                # mask_color = cmap(mask)
                mask_color = color.label2rgb(mask.cpu().numpy())
                mask_color = skimage.img_as_ubyte(mask_color)

                print(img.shape, mask_color.shape)
                save_img = np.hstack((img, mask_color))
                p = path.join(save_dir, f'{t}_{bidx}_{b_i}.jpg')
                print(f'=> {p}')
                imageio.imwrite(p, save_img)
            
