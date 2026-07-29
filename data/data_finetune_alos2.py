# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

# -------------------------------------------------------- 
# Modified by Nevrez Imamoglu (AIST) and Ali Caglayan (AIST) on August 2025
# revisions include changes for:
# * alos2 dataset integration for finetuning instead of imagenet
# * change in transform funbction to accomodate single channel SAR data and ly use of horizontal and vertical flips
# * dataloader builder function for finetuning instead of pretraining and related transform function
# * custom dataloader and dataset class for ALOS2 SAR images to read from csv files with paths for train, val, test splits of the finetuning task
# --------------------------------------------------------

import numpy as np
import rasterio
import csv
import os
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from alos2utils.paired_transform import PairedCompose, PairedRandomHorizontalFlip, PairedRandomVerticalFlip, PairedNormalize
from alos2utils.alos2sm1dataset_finetune import ALOS2SM1Dataset

alos2sm1_mean = [-10.4327]
alos2sm1_std = [5.8041]
ALOS2_DEFAULT_MEAN = alos2sm1_mean
ALOS2_DEFAULT_STD = alos2sm1_std

def build_loader_finetune(config, logger):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(data_mode='train', config=config, logger=logger)

    config.freeze()
    dataset_val, _ = build_dataset(data_mode='val', config=config, logger=logger)

    config.freeze()
    dataset_test, _ = build_dataset(data_mode='test', config=config, logger=logger)

    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}, test images = {len(dataset_test)}")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    sampler_test = DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test


def build_dataset(data_mode, config, logger): #datamode -> 'train', 'val', 'test'
    is_train = True if data_mode == 'train' else False
    if data_mode not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid data mode: {data_mode}. Expected 'train', 'val', or 'test'.")
    transform = build_transform(is_train)
    logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')
    
    if data_mode == 'train':        
        dataset = ALOS2SM1Dataset(csv_file="data/2022_fall_train_no_forest.csv", data_root=config.DATA.DATA_PATH, label_root=config.DATA.LABEL_PATH, transform=transform)
        nb_classes = 9
    elif data_mode == 'val':
        dataset = ALOS2SM1Dataset(csv_file="data/2022_fall_val_no_forest.csv", data_root=config.DATA.DATA_PATH, label_root=config.DATA.LABEL_PATH, transform=transform)
        nb_classes = 9
    elif data_mode == 'test':
        dataset = ALOS2SM1Dataset(csv_file="data/2022_fall_test_no_forest.csv", data_root=config.DATA.DATA_PATH, label_root=config.DATA.LABEL_PATH, transform=transform)
        nb_classes = 9
    else:
        raise NotImplementedError("data_mode is ecpected to be Expected 'train', 'val', or 'test'. Got {data_mode} instead.")
    
    # if config.DATA.DATASET == 'imagenet':
    #     prefix = 'train' if is_train else 'val'
    #     root = os.path.join(config.DATA.DATA_PATH, prefix)
    #     dataset = datasets.ImageFolder(root, transform=transform)
    #     nb_classes = 1000
    # else:
    #     raise NotImplementedError("We only support ImageNet Now.")    

    return dataset, nb_classes


def build_transform(is_train):    
    if is_train:
        transform = PairedCompose([
            PairedRandomHorizontalFlip(),
            PairedRandomVerticalFlip(),
            PairedNormalize(mean=ALOS2_DEFAULT_MEAN, std=ALOS2_DEFAULT_STD)
        ])
    else:
        transform = PairedCompose([
            PairedNormalize(mean=ALOS2_DEFAULT_MEAN, std=ALOS2_DEFAULT_STD)
        ])
    
    return transform
