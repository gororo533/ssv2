"""
Training utilities for VideoMAE.
Includes: cosine scheduler, checkpoint save/load, AverageMeter, logging.
"""
import os
import math
import time
import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CosineScheduler:
    """
    Cosine annealing learning rate scheduler with linear warmup.
    """
    def __init__(self, optimizer, base_lr, min_lr, epochs, warmup_epochs,
                 steps_per_epoch):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.current_step = 0

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def save_checkpoint(state, output_dir, filename='checkpoint.pth'):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    print(f"  → Saved checkpoint: {filepath}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('best_loss', float('inf'))
    print(f"  → Loaded checkpoint from epoch {epoch}")
    return epoch, best_loss


def compute_pretrain_loss(model, videos, masks, device):
    """
    Compute pre-training reconstruction loss (MSE on masked patches).

    Args:
        model: PretrainVisionTransformer
        videos: [B, C, T, H, W]
        masks: [B, N] boolean mask (True = masked)
        device: torch device
    Returns:
        loss: scalar MSE loss
    """
    videos = videos.to(device)
    masks = masks.to(device)

    # Forward pass — predict masked patch pixels
    pred = model(videos, masks)  # [B, N_mask, 3*t*p*p]

    # Get target: original pixel values for masked patches
    with torch.no_grad():
        target = patchify(
            videos,
            model.patch_size,
            model.tubelet_size
        )  # [B, N, 3*t*p*p]

        # Select only masked patches as targets
        B = target.shape[0]
        target = target[masks].reshape(B, -1, target.shape[-1])
        # Normalize target (per-patch normalization as in MAE)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

    loss = torch.nn.functional.mse_loss(pred, target)
    return loss


def patchify(videos, patch_size, tubelet_size):
    """
    Convert video tensor to patch pixels.
    [B, 3, T, H, W] → [B, N, 3 * tubelet * patch^2]
    """
    B, C, T, H, W = videos.shape
    t = T // tubelet_size
    h = H // patch_size
    w = W // patch_size
    p = patch_size
    u = tubelet_size

    # Reshape to extract patches
    x = videos.reshape(B, C, t, u, h, p, w, p)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # [B, t, h, w, C, u, p, p]
    x = x.reshape(B, t * h * w, C * u * p * p)  # [B, N, C*u*p*p]
    return x


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)

    # Filter out invalid labels (label == -1)
    valid = target >= 0
    if valid.sum() == 0:
        return [torch.tensor(0.0) for _ in topk]

    output = output[valid]
    target = target[valid]
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class TensorBoardLogger:
    """Simple TensorBoard logger wrapper."""
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()


def format_time(seconds):
    """Format seconds into human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
