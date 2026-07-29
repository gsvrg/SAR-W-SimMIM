# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
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
import matplotlib.pyplot as plt
#on linux
# try:
#     # noinspection PyUnresolvedReferences
#     from apex import amp
# except ImportError:
#     amp = None
from torch.cuda import amp #on windows
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
    if True:
        resume_file = "ckpt_epoch_100.pth"
        if resume_file:            
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto-resuming from {resume_file}")
        else:
            logger.info(f"No checkpoint found in {config.OUTPUT}, skipping auto-resume")

    
    max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, scaler)

    
    use_amp = (config.AMP_OPT_LEVEL != "O0")

    model.eval()
    for idx, (img, mask, _) in enumerate(data_loader_train):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):#with torch.amp.autocast(enabled=use_amp):
            x, x_rec, mask = model.module.output_simmim(img, mask*0.0)
        
        visualize_simmim_reconstruction(x, x_rec, mask)
        break

    


def visualize_simmim_reconstruction(x, x_rec, mask, save_path="pretrained_simmim_reconstruction.png"):
    """
    Visualize original, reconstructed, and masked regions.
    
    Args:
        x: Original image tensor of shape (1, 1, H, W)
        x_rec: Reconstructed image tensor of shape (1, 1, H, W)
        mask: Binary mask tensor of shape (1, 1, H, W)
        save_path: If provided, saves the figure to this path
    """
    # Ensure tensors are on CPU and convert to numpy
    x_np = x[0].detach().cpu().squeeze().numpy()
    x_rec_np = x_rec[0].detach().cpu().squeeze().numpy()
    mask_np = mask[0].detach().cpu().squeeze().numpy()

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(x_np, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(x_rec_np, cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')

    axs[2].imshow(mask_np, cmap='gray')
    axs[2].set_title('Mask')
    axs[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
        

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
    print(rank)
    print(world_size)
    # #on windows
    # rendezvous_path = "file:///" + os.path.abspath("./tmp_sharedfile")    
    # torch.distributed.init_process_group(backend='gloo', init_method=rendezvous_path, world_size=world_size, rank=rank)
    
    #on linux
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    # Platform-specific process group init
    print(platform.system())
    print(platform.system())
    print(platform.system())
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

    # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
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
