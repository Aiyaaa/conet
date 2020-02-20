import os
import time
from os import path

import numpy as np
import torch
from medpy import metric
from scipy import ndimage
from torch import nn, optim
from torch.nn.parallel import data_parallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# from datasets.brains18 import BrainS18Dataset
from conet.datasets.dataset import DukeOctDataset

from conet.loss_functions.ce_warp import ce_wp
from conet.loss_functions.dice_loss import DC_and_CE_loss
from conet.models.seg_models import FpnSeResnet50, unet
from conet.setting import parse_opts
from conet.trainer.basetrainer import BaseTrainer
from conet.utils.logger import log
from conet.utils.nd_softmax import softmax_helper
from conet.utils.tensor_utilities import sum_tensor


class BengaliTrainer(BaseTrainer):
    def __init__(self, output_folder, sets, model_name=None, max_num_epochs=1000):
        super().__init__()
        self.output_folder = path.join(output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        self.sets = sets
        # self.init_lr = 3e-5
        self.init_lr = 3e-4
        self.weight_decay = 1e-5
        # SET THESE IN self.initialize()
        self.network = unet()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), self.init_lr, weight_decay=self.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                                       cooldown=5, patience=20, min_lr=1e-7,
                                                                       verbose=True)

        self.network.cuda()

        self.max_num_epochs = max_num_epochs

        # dataset

        training_dataset = DukeOctFlatDataset(split='train')
        val_dataset = DukeOctFlatDataset(split='val')

        data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True,
                                 num_workers=sets.num_workers, pin_memory=sets.pin_memory)
        val_loader = DataLoader(val_dataset, sets.batch_size, shuffle=True, num_workers=sets.num_workers,
                                pin_memory=sets.pin_memory)
        self.tr_gen = data_loader
        self.val_gen = val_loader

        self.was_initialized = True
        ################# SET THESE IN INIT ################################################

    def train(self):
        pass

    def run_iteration(self, data_dict, do_backprop=True, run_online_evaluation=False):
        # data_dict = next(data_generator)
        data = data_dict['image']
        target = data_dict['mask']

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # soft_label_target = soft_label_target.cuda(non_blocking=True)

        self.optimizer.zero_grad()

        # output = self.network(data)
        output = data_parallel(self.network, data)

        del data
        ce_dice_loss = self.loss(output, target)

        kl_loss = 0.
        l = ce_dice_loss + kl_loss

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        if do_backprop:
            l.backward()
            self.optimizer.step()

        return {
            'loss': l.detach().cpu().numpy()
        }

    def validate(self):
        self.network.eval()
        from skimage.color import label2rgb
        import skimage
        from skimage.io import imsave
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        save_dir = path.join(self.output_folder, 'validate')
        save_gt_dir = path.join(save_dir, 'gt')

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_gt_dir, exist_ok=True)

        for data_dict in self.val_gen:
            data = data_dict['image']
            label = data_dict['mask']

            if not isinstance(data, torch.Tensor):
                data = torch.from_numpy(data).float()
            data = data.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            # soft_label_target = soft_label_target.cuda(non_blocking=True)
            # output = self.network(data)
            output = data_parallel(self.network, data)
            output = torch.softmax(output, dim=1)
            pred_labels = torch.argmax(output, dim=1)
            for img, truth, pred_label, fname in zip(data, label, pred_labels, data_dict['fname']):
                img = img.cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = img * std + mean
                img = skimage.img_as_ubyte(img)
                t_label = skimage.img_as_ubyte(label2rgb(truth.cpu().numpy()))
                rgb_label = skimage.img_as_ubyte(
                    label2rgb(pred_label.cpu().numpy()))
                bbb_name = path.basename(fname).split('.')[0]
                save_fname = path.basename(fname).split('.')[0] + '.jpg'
                save_fname = path.join(save_dir, save_fname)

                save_im = np.hstack([img, t_label, rgb_label])
                imsave(save_fname, save_im)

                # logit = logit.cpu().numpy()
                save_np = save_fname.replace('.jpg', '.npy')
                np.save(save_np, pred_label.cpu().numpy())

                # gt
                save_gt = path.join(save_gt_dir, f'{bbb_name}.npy')
                np.save(save_gt, truth.cpu().numpy())

    def run_online_evaluation(self, output, target):
        pass

    def finish_online_evaluation(self):
        pass
