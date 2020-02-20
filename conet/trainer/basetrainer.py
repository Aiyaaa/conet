import os
import random
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from os import path
from time import sleep, time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from torch import nn, optim
from torch.nn.parallel import data_parallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from conet.utils.logger import log
from tensorboardX import SummaryWriter

matplotlib.use("agg")


def maybe_mkdir_p(dir):
    os.makedirs(dir, exist_ok=True)


class BaseTrainer:
    def __init__(self, output_folder, deterministic=False, max_num_epochs=1000):
        # random.seed(42)
        # np.random.seed(42)
        # torch.manual_seed(42)
        # torch.cuda.manual_seed_all(42)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        ################# SET THESE IN self.initialize() ###################################
        self.network = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.was_initialized = False

        ################# SET THESE IN INIT ################################################
        self.output_folder = output_folder
        self.epoch = 0

        self.max_num_epochs = max_num_epochs

        self.log_file = None
        self.loss = None

        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.tr_log_items = defaultdict(list)
        self.val_log_items = defaultdict(list)
        self.val_metrics = defaultdict(list)

        self.epoch = 0

        self.deterministic = deterministic

        self.was_initialized = False

        self.board_writer = SummaryWriter(self.output_folder)

    def run_training(self):
        torch.cuda.empty_cache()
        bst_val_loss = None
        bst_val_metric = None
        while self.epoch < self.max_num_epochs:

            self.print_to_log_file("\nepoch: ", self.epoch)

            train_losses_epoch = defaultdict(list)

            # train one epoch
            self.network.train()
            tbar = tqdm(self.tr_gen, ascii=True)
            for b in tbar:
                log_bags = self.run_iteration(b, True)
                l = log_bags['loss']
                tbar.set_description(f'train loss: {l:.5f}')
                for loss_name, tr_loss in log_bags.items():
                    train_losses_epoch[loss_name].append(tr_loss)

            for loss_name, loss_epoch in train_losses_epoch.items():
                l_e = np.mean(loss_epoch)
                self.board_writer.add_scalar(
                    f'tr/{loss_name}', l_e, self.epoch)
                self.tr_log_items[loss_name].append(l_e)

            log_message = ' '.join(
                [f'{loss_name}: {self.tr_log_items[loss_name][-1]:.4f}' for loss_name in self.tr_log_items.keys()])
            self.print_to_log_file(f"train {log_message}")

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses_epoch = defaultdict(list)
                tbar = tqdm(self.val_gen, ascii=True)
                for b in tbar:
                    log_bags = self.run_iteration(b, False, True)
                    l = log_bags['loss']
                    tbar.set_description(f'val loss: {l:.5f}')
                    for loss_name, val_loss in log_bags.items():
                        val_losses_epoch[loss_name].append(val_loss)

            for loss_name, loss_epoch in val_losses_epoch.items():
                l_e = np.mean(loss_epoch)
                self.board_writer.add_scalar(
                    f'val/{loss_name}', l_e, self.epoch)
                self.val_log_items[loss_name].append(l_e)

            self.on_epoch_end()
            if bst_val_metric is None:
                bst_val_metric = self.all_val_eval_metrics[-1]
                bst_val_metric = self.val_metrics['val_metric'][-1]

            if bst_val_metric < self.val_metrics['val_metric'][-1]:
                bst_val_metric = self.val_metrics['val_metric'][-1]

                self.save_checkpoint(
                    path.join(self.output_folder, "model_best.model"))

            self.lr_scheduler.step(np.mean(val_losses_epoch['loss']))
            # self.lr_scheduler.step(self.val_metrics['bst'][-1])
            self.print_to_log_file(
                "val loss (train=False): %.4f" % self.val_log_items['loss'][-1])

            self.epoch += 1
            # self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))

        self.save_checkpoint(path.join(self.output_folder,
                                       "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if path.isfile(path.join(self.output_folder, "model_latest.model")):
            os.remove(path.join(self.output_folder, "model_latest.model"))
        if path.isfile(path.join(self.output_folder, "model_latest.model.pkl")):
            os.remove(path.join(self.output_folder, "model_latest.model.pkl"))

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            for key in lr_sched_state_dct.keys():
                lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        torch.save({
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.tr_log_items, self.val_log_items, self.val_metrics)},
            fname)
        self.print_to_log_file(
            "done, saving took %.2f seconds" % (time() - start_time))

    def load_best_checkpoint(self, train=True):
        self.load_checkpoint(path.join(self.output_folder,
                                       "model_best.model"), train=train)

    def load_latest_checkpoint(self, train=True):
        if path.isfile(path.join(self.output_folder, "model_final_checkpoint.model")):
            return self.load_checkpoint(path.join(self.output_folder, "model_final_checkpoint.model"), train=train)
        if path.isfile(path.join(self.output_folder, "model_latest.model")):
            return self.load_checkpoint(path.join(self.output_folder, "model_latest.model"), train=train)
        all_checkpoints = [i for i in os.listdir(
            self.output_folder) if i.endswith(".model") and i.find("_ep_") != -1]
        if len(all_checkpoints) == 0:
            return self.load_best_checkpoint(train=train)
        corresponding_epochs = [int(i.split("_")[-1].split(".")[0])
                                for i in all_checkpoints]
        checkpoint = all_checkpoints[np.argmax(corresponding_epochs)]
        self.load_checkpoint(
            path.join(self.output_folder, checkpoint), train=train)

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        saved_model = torch.load(fname, map_location=torch.device(
            'cuda', torch.cuda.current_device()))
        self.load_checkpoint_ram(saved_model, train)

    def load_checkpoint_ram(self, saved_model, train=True):
        """
        used for if the checkpoint is already in ram
        :param saved_model:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in saved_model['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys:
                key = key[7:]
            new_state_dict[key] = value
        self.network.load_state_dict(new_state_dict)
        self.epoch = saved_model['epoch']
        print(f'Load from epoch: {self.epoch}')
        if train:
            optimizer_state_dict = saved_model['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.load_state_dict(
                    saved_model['lr_scheduler_state_dict'])
        self.tr_log_items, self.val_log_items, self.val_metrics = saved_model[
            'plot_stuff']

    def run_iteration(self, data_dict, do_backprop=True, run_online_evaluation=False):
        raise NotImplementedError

    def validate(self, train=True):
        pass

    def initialize(self):
        pass

    def log_val_metric(self):
        for metric_name, metric_epoch in self.val_metrics.items():
            self.board_writer.add_scalar(
                f'metric/{metric_name}', metric_epoch, self.epoch)

    def on_epoch_end(self):
        self.finish_online_evaluation()
        self.plot_progress()

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            colors = "bgrcmykw"
            color_index = 0

            x_values = list(range(self.epoch + 1))

            for loss_name in self.tr_log_items.keys():
                ax.plot(x_values, self.tr_log_items[loss_name], color=colors[color_index], ls='-',
                        label=f"{loss_name}_tr")
                color_index += 1

            for loss_name in self.val_log_items.keys():
                ax.plot(x_values, self.val_log_items[loss_name], color=colors[color_index], ls='-',
                        label=f"{loss_name}_val")
                color_index += 1

            for metric_name in self.val_metrics.keys():
                ax2.plot(x_values, self.val_metrics[metric_name], color=colors[color_index], ls='--',
                         label=f"{metric_name}")
                color_index += 1

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(path.join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = path.join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                      (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                       timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " %
                      datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def run_online_evaluation(self, *args, **kwargs):
        """
        Can be implemented, does not have to
        :param output_torch:
        :param target_npy:
        :return:
        """
        pass

    def finish_online_evaluation(self, *args, **kwargs):
        pass
