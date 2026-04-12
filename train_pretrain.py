"""
VideoMAE Pre-training Script (Stage 1).
Self-supervised masked video reconstruction on SSv2.

Usage:
    python train_pretrain.py --config configs/videomae_tiny_ssv2.yaml
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from models.videomae import build_pretrain_model
from dataset.ssv2_dataset import build_pretraining_dataset
from utils.train_utils import (
    load_config, AverageMeter, CosineScheduler,
    save_checkpoint, compute_pretrain_loss,
    TensorBoardLogger, format_time,
)


def parse_args():
    parser = argparse.ArgumentParser('VideoMAE Pre-training')
    parser.add_argument('--config', type=str,
                        default='configs/videomae_tiny_ssv2.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler,
                    device, epoch, config, logger, global_step):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    grad_accum = config['pretrain']['gradient_accumulation']
    log_freq = config['pretrain']['log_freq']
    use_amp = config['pretrain']['use_amp'] and device.type == 'cuda'

    optimizer.zero_grad()
    start_time = time.time()

    for step, (videos, masks) in enumerate(dataloader):
        videos = videos.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Mixed precision forward
        if use_amp:
            with autocast('cuda'):
                loss = compute_pretrain_loss(model, videos, masks, device)
                loss = loss / grad_accum
        else:
            loss = compute_pretrain_loss(model, videos, masks, device)
            loss = loss / grad_accum

        # Backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (step + 1) % grad_accum == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            lr = scheduler.step()
            global_step += 1

            # Log
            if logger:
                logger.log_scalar('pretrain/loss', loss.item() * grad_accum,
                                  global_step)
                logger.log_scalar('pretrain/lr', lr, global_step)

        loss_meter.update(loss.item() * grad_accum, videos.size(0))

        if (step + 1) % log_freq == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (step + 1) * (len(dataloader) - step - 1)
            print(f"  Epoch [{epoch}] Step [{step+1}/{len(dataloader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"LR: {scheduler.get_lr():.2e} "
                  f"ETA: {format_time(eta)}")

    return loss_meter.avg, global_step


def main():
    args = parse_args()
    config = load_config(args.config)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Build dataset & dataloader
    print("\nBuilding dataset...")
    dataset = build_pretraining_dataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['pretrain']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config['data']['num_workers'] > 0 else False,
    )

    # Build model
    print("\nBuilding model...")
    model = build_pretrain_model(config)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params / 1e6:.2f}M")
    print(f"  Trainable params: {n_trainable / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['pretrain']['lr'],
        weight_decay=config['pretrain']['weight_decay'],
        betas=(0.9, 0.95),
    )

    # Scheduler
    steps_per_epoch = len(dataloader) // config['pretrain']['gradient_accumulation']
    scheduler = CosineScheduler(
        optimizer,
        base_lr=config['pretrain']['lr'],
        min_lr=config['pretrain']['min_lr'],
        epochs=config['pretrain']['epochs'],
        warmup_epochs=config['pretrain']['warmup_epochs'],
        steps_per_epoch=max(1, steps_per_epoch),
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda', enabled=config['pretrain']['use_amp'] and device.type == 'cuda')

    # Logger
    log_dir = os.path.join(config['pretrain']['output_dir'], 'logs')
    logger = TensorBoardLogger(log_dir)

    # Resume
    start_epoch = 0
    best_loss = float('inf')
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        global_step = ckpt.get('global_step', 0)
        print(f"  → Resumed from epoch {start_epoch}")

    # Training loop
    print(f"\nStarting pre-training for {config['pretrain']['epochs']} epochs")
    print(f"   Batch size: {config['pretrain']['batch_size']} "
          f"× {config['pretrain']['gradient_accumulation']} grad accum "
          f"= {config['pretrain']['batch_size'] * config['pretrain']['gradient_accumulation']} effective")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Steps per epoch: {len(dataloader)}")
    print()

    for epoch in range(start_epoch, config['pretrain']['epochs']):
        epoch_start = time.time()

        loss, global_step = train_one_epoch(
            model, dataloader, optimizer, scheduler, scaler,
            device, epoch, config, logger, global_step)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {format_time(epoch_time)} | "
              f"Loss: {loss:.4f}")

        # Save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if (epoch + 1) % config['pretrain']['save_freq'] == 0 or is_best:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'global_step': global_step,
                'config': config,
            }
            save_checkpoint(state, config['pretrain']['output_dir'],
                            f'checkpoint_epoch{epoch}.pth')
            if is_best:
                save_checkpoint(state, config['pretrain']['output_dir'],
                                'checkpoint_best.pth')

    logger.close()
    print(f"\nPre-training complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
