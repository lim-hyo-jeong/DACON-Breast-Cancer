import numpy as np
import torch
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
import os
import random


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_logger(filename):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    log_formatter = Formatter(
        '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler1.setFormatter(log_formatter)
    handler2 = FileHandler(filename=filename)
    handler2.setFormatter(log_formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_scheduler(optimizer, training_params):
    scheduler_params = training_params['scheduler_params']

    if training_params['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params['T_0'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1
        )

    elif training_params['scheduler'] == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=scheduler_params['max_lr'],
            steps_per_epochs=scheduler_params['steps_per_epochs'],
            epochs=training_params['epochs'],
        )

    elif training_params['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params['T_max'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1,
        )

    elif training_params['scheduler'] == 'ReduceLROnPlateau':
        sch_mode = 'min' if training_params['monitor'] == 'loss' else 'max'
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=sch_mode,
            factor=scheduler_params['factor'],
            patience=scheduler_params['patience'],
            min_lr=scheduler_params['min_lr'],
            verbose=True)

    return scheduler
