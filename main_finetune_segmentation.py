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
# * segmentation task instead of classification and related  chages such as evaluation metrics
# * changes in loss function to use combined loss (dice + focal)
# * changes in dataset loading to accomodate ALOS2 data
# * changes in visualization to save segmentation results
# * changes in configuration to add loss parameters
# * changes in logging to log segmentation metrics
# * changes in checkpointing to save best model based on mIoU
# * changes in command line arguments to add loss parameters
# * changes in data loading to use alos2utils
# * changes in optimizer to accomodate segmentation task
# * adding evaluate_segmentation function to compute mIoU, mAcc, PixelAcc for validation and test sets
# * adding compute_iou_from_confusion_matrix and compute_acc_from_confusion_matrix helper functions
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from alos2utils.losses import CombinedLoss
from alos2utils.visuals import save_batch_samples_visualizations, class_palette, class_names
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
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
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
    parser.add_argument('--label-path', type=str, help='path to label images')
    parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    # Loss params
    parser.add_argument('--alpha_scale', type=float, default=0.5, metavar='SCALAR', help='Scaling factor for alpha in focal loss (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='GAMMA', help='Focusing parameter for focal loss (default: 2.0)')
    parser.add_argument('--lambda_dice', type=float, default=0.5, metavar='LDICE', help='Weight for the Dice loss component in the combined loss (default: 0.5)')
    parser.add_argument('--lambda_focal', type=float, default=0.5, metavar='LFOCAL', help='Weight for the Focal loss component in the combined loss (default: 0.5)')


    # distributed training
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0), help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)
    config.defrost()
    config.LOSS_PARAMS.SCALAR = args.alpha_scale
    config.LOSS_PARAMS.GAMMA = args.gamma
    config.LOSS_PARAMS.LDICE = args.lambda_dice
    config.LOSS_PARAMS.LFOCAL = args.lambda_focal
    config.freeze()

    return args, config


def main(config):
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(config, logger, is_pretrain=False)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=False)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    # if config.AMP_OPT_LEVEL != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    
    # Native PyTorch AMP (automatic mixed precision)
    scaler = torch.amp.GradScaler('cuda') if config.AMP_OPT_LEVEL != "O0" else None
    
    # move model to GPU first
    model = model.to(config.LOCAL_RANK)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], output_device=config.LOCAL_RANK, broadcast_buffers=False, find_unused_parameters=True)
    
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    
    weights = torch.tensor([0.11347487, 0.10774269, 0.10684644, 0.11319541, 0.06752731, 0.12183155, 0.12457897, 0.12064208, 0.12416070])
    weights = weights.to('cuda')
    # Use Combined Loss (50% Dice Loss, 50% Focal Loss)
    criterion = CombinedLoss(alpha=weights, alpha_scale=config.LOSS_PARAMS.SCALAR, gamma= config.LOSS_PARAMS.GAMMA , lambda_dice=config.LOSS_PARAMS.LDICE, lambda_focal=config.LOSS_PARAMS.LFOCAL, ignore_index=-1)

    max_accuracy = 0.0
    max_mIoU = 0.0  

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger,scaler)
        val_metrics = evaluate_segmentation(
                model, data_loader_val, criterion, 0, device="cuda",
                num_classes=config.MODEL.NUM_CLASSES,
                ignore_index=config.DATA.IGNORE_INDEX
            )
        logger.info(f"Validation mIoU={val_metrics['mIoU']:.4f}")
        if config.EVAL_MODE:
            return
    elif config.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, scaler)
        # if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        #     save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, scaler)

        # === Evaluate on validation set ===
        val_metrics = evaluate_segmentation(
            model, data_loader_val, criterion, epoch, device="cuda", num_classes=config.MODEL.NUM_CLASSES,
            ignore_index=config.DATA.IGNORE_INDEX, is_test=False
        )

        val_mIoU = val_metrics["mIoU"]
        logger.info(f"[Epoch {epoch}] Val mIoU={val_mIoU:.4f}, "
                    f"mAcc={val_metrics['mAcc']:.4f}, "
                    f"PixelAcc={val_metrics['pixel_accuracy']:.4f}, "
                    f"Loss={val_metrics['loss']:.4f}")
        
        # === Save best model ===
        if dist.get_rank() == 0:
            if val_mIoU > max_mIoU:
                logger.info(f"New best model at epoch {epoch}! (mIoU {val_mIoU:.4f} > {max_mIoU:.4f})")
                max_mIoU = val_mIoU
                max_accuracy = val_mIoU
                save_checkpoint(
                    config, epoch, model_without_ddp, max_accuracy,
                    optimizer, lr_scheduler, logger, scaler
                )
                if epoch > -1: #config.TRAIN.WARMUP_EPOCHS:                    
                    # === Evaluate on test set ===
                    test_metrics = evaluate_segmentation(
                        model, data_loader_test, criterion, epoch, device="cuda", num_classes=config.MODEL.NUM_CLASSES,
                        ignore_index=config.DATA.IGNORE_INDEX, is_test=True
                    )

                    test_mIoU = test_metrics["mIoU"]
                    logger.info(f"[Epoch {epoch}] Test mIoU={test_mIoU:.4f}, "
                                f"mAcc={test_metrics['mAcc']:.4f}, "
                                f"PixelAcc={test_metrics['pixel_accuracy']:.4f}, "
                                f"Loss={test_metrics['loss']:.4f}")                                                      

            # # Optionally still save periodic checkpoints
            # if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            #     save_checkpoint(
            #         config, epoch, model_without_ddp, max_mIoU,
            #         optimizer, lr_scheduler, logger, scaler
            #     )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, scaler):
    model.train()
    optimizer.zero_grad()

    logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    accumulation_steps = config.TRAIN.ACCUMULATION_STEPS
    clip_grad = config.TRAIN.CLIP_GRAD
    use_amp = (config.AMP_OPT_LEVEL != "O0")

    for idx, (samples, targets, _) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).long()

        # Forward pass with autocast
        with torch.amp.autocast('cuda',enabled=use_amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)

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

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[-1]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            # if idx > 450:
            #     break

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")



@torch.no_grad()
def evaluate_segmentation(model, dataloader, criterion, epoch, device, num_classes, ignore_index=-1, is_test=False):
    """
    Evaluate a segmentation model on a dataloader.

    Returns:
        dict with metrics: loss, pixel_accuracy, mIoU, mAcc, per_class_iou, per_class_acc
    """
    model.eval()
    
    total_loss = 0.0
    total_pixels = 0
    total_correct = 0
    use_amp = (config.AMP_OPT_LEVEL != "O0")

    # Confusion matrix (num_classes x num_classes)
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)

    with torch.no_grad():
        for idx, (images, targets, *_) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()

            with torch.autocast("cuda",enabled=use_amp):
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):  # in case model returns multi-scale outputs
                    outputs = outputs[0]
                loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            # ---- Valid mask for ignore_index ----
            valid_mask = (targets != ignore_index)
            preds_valid = preds[valid_mask]
            targets_valid = targets[valid_mask]

            # Pixel accuracy
            total_correct += (preds_valid == targets_valid).sum().item()
            total_pixels += valid_mask.sum().item()

            # Update confusion matrix
            indices = torch.stack([targets_valid, preds_valid], dim=0)
            cm = torch.zeros(num_classes, num_classes, device=confusion_matrix.device)
            cm.index_put_(tuple(indices), torch.ones_like(targets_valid, dtype=torch.float), accumulate=True)
            confusion_matrix += cm
        
    if is_test:
        # Save some visualizations of test set predictions
        save_dir = os.path.join(config.OUTPUT, 'test_visualizations')
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving test set visualizations to {save_dir}")
            save_batch_samples_visualizations(
                images,
                preds,
                targets,
                epoch,
                idx,
                save_dir,
                class_palette,
                class_names,
                max_samples=8
            )

    # ===== Metrics =====
    per_class_iou = compute_iou_from_confusion_matrix(confusion_matrix)
    per_class_acc = compute_acc_from_confusion_matrix(confusion_matrix)

    mIoU = torch.nanmean(per_class_iou).item()
    mAcc = torch.nanmean(per_class_acc).item()
    pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    avg_loss = total_loss / len(dataloader.dataset)

    print(f"mIoU {mIoU:.4f} mAcc {mAcc:.4f} "
          f"Pixel Accuracy {pixel_accuracy:.4f} "
          f"Loss {avg_loss:.4f}")

    return {
        "loss": avg_loss,
        "pixel_accuracy": pixel_accuracy,
        "mIoU": mIoU,
        "mAcc": mAcc,
        "per_class_iou": per_class_iou.cpu().numpy().tolist(),
        "per_class_acc": per_class_acc.cpu().numpy().tolist(),
    }


def compute_iou_from_confusion_matrix(confusion_matrix):
    """IoU = TP / (TP + FP + FN)"""
    diag = torch.diag(confusion_matrix)
    denom = (confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - diag)
    iou = diag / torch.clamp(denom, min=1e-6)
    return iou


def compute_acc_from_confusion_matrix(confusion_matrix):
    """Class accuracy = TP / (TP + FN)"""
    diag = torch.diag(confusion_matrix)
    row_sum = confusion_matrix.sum(dim=1)
    acc = diag / torch.clamp(row_sum, min=1e-6)
    return acc


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


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
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()

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
