"""
Evaluation script for fine-tuned VideoMAE model.
Reports Top-1 and Top-5 accuracy on the validation set.

Usage:
    python evaluate.py --config configs/videomae_tiny_ssv2.yaml --checkpoint output/finetune/checkpoint_best.pth
"""
import os
import argparse
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from collections import defaultdict

from models.videomae import build_finetune_model
from dataset.ssv2_dataset import build_dataset
from utils.train_utils import load_config, AverageMeter, accuracy, format_time


def parse_args():
    parser = argparse.ArgumentParser('VideoMAE Evaluation')
    parser.add_argument('--config', type=str,
                        default='configs/videomae_tiny_ssv2.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned checkpoint')
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=True):
    """Full evaluation on validation set."""
    model.eval()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    all_preds = []
    all_labels = []

    start_time = time.time()
    for step, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        valid = labels >= 0
        if valid.sum() == 0:
            continue
        videos = videos[valid]
        labels = labels[valid]

        if use_amp and device.type == 'cuda':
            with autocast('cuda'):
                logits = model(videos)
        else:
            logits = model(videos)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        acc1_meter.update(acc1.item(), videos.size(0))
        acc5_meter.update(acc5.item(), videos.size(0))

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [{step+1}/{len(dataloader)}] "
                  f"Acc@1: {acc1_meter.avg:.1f}% Acc@5: {acc5_meter.avg:.1f}% "
                  f"({format_time(elapsed)})")

    total_time = time.time() - start_time
    return acc1_meter.avg, acc5_meter.avg, all_preds, all_labels, total_time


def per_class_accuracy(preds, labels, num_classes):
    """Compute per-class accuracy."""
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for p, l in zip(preds, labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    per_class = {}
    for c in range(num_classes):
        total = class_total.get(c, 0)
        correct = class_correct.get(c, 0)
        per_class[c] = correct / total if total > 0 else 0.0

    return per_class


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build dataset
    print("\nBuilding validation dataset...")
    val_dataset = build_dataset(config, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )

    # Build & load model
    print("\nBuilding model...")
    model = build_finetune_model(config)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    print(f"  Loaded checkpoint: {args.checkpoint}")
    print(f"  From epoch: {ckpt.get('epoch', '?')}")

    # Evaluate
    print(f"\nEvaluating on {len(val_dataset)} validation samples...")
    acc1, acc5, preds, labels, elapsed = evaluate(
        model, val_loader, device,
        use_amp=config['finetune']['use_amp'])

    print(f"\n{'='*50}")
    print(f"  Top-1 Accuracy: {acc1:.2f}%")
    print(f"  Top-5 Accuracy: {acc5:.2f}%")
    print(f"  Total samples:  {len(preds)}")
    print(f"  Eval time:      {format_time(elapsed)}")
    print(f"{'='*50}")

    # Per-class accuracy
    num_classes = config['model']['num_classes']
    pc_acc = per_class_accuracy(preds, labels, num_classes)

    # Load label names
    labels_json = config['data']['labels_json']
    with open(labels_json, 'r') as f:
        label_map = json.load(f)
    id_to_name = {int(v): k for k, v in label_map.items()}

    # Print top/bottom 5 classes
    sorted_classes = sorted(pc_acc.items(), key=lambda x: x[1], reverse=True)
    non_zero = [(c, a) for c, a in sorted_classes if a > 0]

    if non_zero:
        print(f"\nTop 5 classes:")
        for c, a in non_zero[:5]:
            name = id_to_name.get(c, f"class_{c}")
            print(f"  {c:3d} ({a*100:.1f}%) {name}")

        print(f"\nBottom 5 classes:")
        for c, a in non_zero[-5:]:
            name = id_to_name.get(c, f"class_{c}")
            print(f"  {c:3d} ({a*100:.1f}%) {name}")

    # Save results
    output_dir = config['finetune']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'acc1': acc1,
        'acc5': acc5,
        'num_samples': len(preds),
        'checkpoint': args.checkpoint,
    }
    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
