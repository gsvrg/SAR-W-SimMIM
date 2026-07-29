# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

# -------------------------------------------------------- 
# Modified by Nevrez Imamoglu (AIST) and Ali Caglayan (AIST) on August 2025
# revisions include changes for:
# * using native PyTorch AMP instead of Nvidia apex
# * platform aware useage of torch.distributed.init_process_group both for linux and windows
# * SAR based weighted loss function support
# --------------------------------------------------------



import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

#on linux
# try:
#     # noinspection PyUnresolvedReferences
#     from apex import amp
# except ImportError:
#     amp = None
from torch.cuda import amp
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import platform

def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0), help='local rank for DistributedDataParallel')


    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    # Initialize distributed training
    # init_distributed_mode(config)

    logger.info(f"Creating data loader...")
    data_loader_train = build_loader(config, logger, is_pretrain=True)

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)

    # Native PyTorch AMP (automatic mixed precision)
    scaler = torch.cuda.amp.GradScaler() if config.AMP_OPT_LEVEL != "O0" else None

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # Auto-resume
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"Auto-resume: overriding resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto-resuming from {resume_file}")
        else:
            logger.info(f"No checkpoint found in {config.OUTPUT}, skipping auto-resume")

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, scaler)

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config=config,
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            scaler=scaler,  # pass scaler if AMP is enabled
            logger = logger
        )

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0.0, optimizer, lr_scheduler, logger, scaler)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, scaler, logger):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    accumulation_steps = config.TRAIN.ACCUMULATION_STEPS
    clip_grad = config.TRAIN.CLIP_GRAD
    use_amp = (config.AMP_OPT_LEVEL != "O0")

    for idx, (img, mask, mse_linear_weight) in enumerate(data_loader):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)        
        if config.MODEL.IS_WEIGHTED_LOSS_FN:
            mse_linear_weight = mse_linear_weight.cuda(non_blocking=True)
        else:
            mse_linear_weight = None

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(img, mask, mse_linear_weight=mse_linear_weight)
        
        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (idx + 1) % accumulation_steps == 0:
            if clip_grad:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            else:
                grad_norm = get_grad_norm(model.parameters())

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            grad_norm = 0.0  # No optimizer step this batch, no meaningful grad norm

        torch.cuda.synchronize()

        loss_meter.update(loss.item() * accumulation_steps, img.size(0))  # multiply back to original loss scale
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    
    #on linux
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # Platform-specific process group init
    print(platform.system())
    if platform.system() == 'Windows':
        rendezvous_file = os.path.abspath("./tmp_sharedfile")
        if not os.path.exists(rendezvous_file):
            with open(rendezvous_file, "w") as f:
                pass  # create empty file

        rendezvous_path = "file:///" + rendezvous_file.replace("\\", "/")
        # Use gloo with file-based rendezvous on Windows
        # rendezvous_path = "file:///" + os.path.abspath("./tmp_sharedfile")

        dist.init_process_group(backend='gloo', init_method=rendezvous_path,
                                world_size=world_size, rank=rank)
    else:
        # Use NCCL with env:// on Linux
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)

    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
