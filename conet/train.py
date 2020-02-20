'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

import importlib
import os
import pkgutil
from os import path
import numpy as np
import torch

from scipy import ndimage
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from datetime import datetime

import conet
from conet.setting import parse_opts
from conet.trainer.trainer import Trainer
from conet.utils.backup import backup_project_as_zip


def recursive_find_trainer(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_trainer(
                    [path.join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    # train from resume
    if sets.resume_path:
        pass

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True
    network_trainer = sets.network_trainer
    print(conet.trainer.__path__)
    search_in = (list(conet.trainer.__path__)[0],)
    base_module = 'conet.trainer'
    trainer_class = recursive_find_trainer([path.join(*search_in)], network_trainer,
                                           current_module=base_module)

    # training
    trainer = trainer_class()
    if sets.model is None or len(sets.model) == 0:
        model_name = None
    else:
        model_name = sets.model
    trainer.initialize(output_folder=sets.save_folder,
                       sets=sets, model_name=model_name)

    # PROJECT_PATH = os.path.dirname(os.path.realpath(os.path.join(__file__, os.path.pardir)))
    PROJECT_PATH = os.path.dirname(os.path.realpath(os.path.join(__file__)))
    IDENTIFIER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # backup_project_as_zip(PROJECT_PATH, os.path.join(sets.save_folder, 'code.train.%s.zip'%IDENTIFIER))

    if sets.continue_run:
        trainer.load_latest_checkpoint()

    if not sets.validation_only:
        trainer.run_training()

    trainer.load_best_checkpoint()
    trainer.validate()
