"""
Smoke test: verify data loading and model forward pass.
Run this before starting full training.

Usage:
    python smoke_test.py
"""
import sys
import time
import torch
import yaml
import numpy as np

def test_model():
    """Test model construction and forward pass."""
    print("=" * 60)
    print("Testing Model Construction")
    print("=" * 60)

    from models.videomae import build_pretrain_model, build_finetune_model
    from utils.train_utils import load_config

    config = load_config('configs/videomae_tiny_ssv2.yaml')
    cfg = config['model']

    # Pre-train model
    print("\n  Building PretrainVisionTransformer...")
    pretrain_model = build_pretrain_model(config)
    n_params = sum(p.numel() for p in pretrain_model.parameters())
    print(f"  Pretrain model: {n_params / 1e6:.2f}M params")

    # Finetune model
    print("\n  Building VisionTransformerForFinetune...")
    finetune_model = build_finetune_model(config)
    n_params = sum(p.numel() for p in finetune_model.parameters())
    print(f"  Finetune model: {n_params / 1e6:.2f}M params")

    # Test forward pass
    print("\n  Testing forward pass...")
    B = 2
    T = cfg['num_frames']
    H = W = cfg['img_size']
    dummy_video = torch.randn(B, 3, T, H, W)

    # Pretrain forward
    t_patches = T // cfg['tubelet_size']
    h_patches = H // cfg['patch_size']
    w_patches = W // cfg['patch_size']
    N = t_patches * h_patches * h_patches
    num_masks = int(config['masking']['ratio'] * h_patches * w_patches)
    num_vis = h_patches * w_patches - num_masks

    mask_per_frame = np.hstack([
        np.zeros(h_patches * w_patches - num_masks),
        np.ones(num_masks),
    ])
    np.random.shuffle(mask_per_frame)
    mask = np.tile(mask_per_frame, (t_patches, 1)).flatten()
    mask = torch.from_numpy(mask).bool().unsqueeze(0).expand(B, -1)

    print(f"  Input video: {dummy_video.shape}")
    print(f"  Total patches: {N}, Visible: {num_vis * t_patches}, "
          f"Masked: {num_masks * t_patches}")

    with torch.no_grad():
        pred = pretrain_model(dummy_video, mask)
    print(f"  Pretrain output: {pred.shape} "
          f"(expected [{B}, {num_masks * t_patches}, "
          f"{3 * cfg['tubelet_size'] * cfg['patch_size']**2}])")

    # Finetune forward
    with torch.no_grad():
        logits = finetune_model(dummy_video)
    print(f"  Finetune output: {logits.shape} "
          f"(expected [{B}, {cfg['num_classes']}])")

    # Memory estimate on GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        pretrain_model = pretrain_model.to(device)
        dummy_video = dummy_video.to(device)
        mask = mask.to(device)

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred = pretrain_model(dummy_video, mask)

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n  GPU peak memory (inference, fp16): {peak_mem:.1f} MB")

        # Also test with gradient
        torch.cuda.reset_peak_memory_stats()
        pretrain_model.train()
        with torch.amp.autocast('cuda'):
            pred = pretrain_model(dummy_video, mask)
            loss = pred.mean()
        loss.backward()

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU peak memory (training, fp16): {peak_mem:.1f} MB")
        print(f"  {'Fits in 2GB!' if peak_mem < 1800 else '⚠️ Might be tight on 2GB'}")

    return True


def test_dataset():
    """Test dataset loading."""
    print("\n" + "=" * 60)
    print("  Testing Dataset Loading")
    print("=" * 60)

    from utils.train_utils import load_config
    from dataset.ssv2_dataset import build_pretraining_dataset, build_dataset

    config = load_config('configs/videomae_tiny_ssv2.yaml')

    # Test pretrain dataset (load just 1 sample)
    print("\n  Building pretrain dataset (this loads metadata only)...")
    pretrain_ds = build_pretraining_dataset(config)

    print(f"  Dataset size: {len(pretrain_ds)}")
    print(f"  Loading first sample (video decoding)...")

    t0 = time.time()
    video, mask = pretrain_ds[0]
    t1 = time.time()

    print(f"  Video tensor: {video.shape} (dtype={video.dtype})")
    print(f"  Mask tensor:  {mask.shape} (dtype={mask.dtype})")
    print(f"  Mask ratio:   {mask.float().mean():.2f} "
          f"({mask.sum()}/{mask.numel()})")
    print(f"   Load time:    {t1 - t0:.2f}s")

    # Test finetune dataset
    print("\n  Building finetune dataset...")
    finetune_ds = build_dataset(config, mode='train')
    video, label = finetune_ds[0]
    print(f"  Video tensor: {video.shape} | Label: {label}")

    val_ds = build_dataset(config, mode='val')
    video, label = val_ds[0]
    print(f"  Val tensor:   {video.shape} | Label: {label}")

    return True


def test_dataloader():
    """Test DataLoader batching."""
    print("\n" + "=" * 60)
    print(" Testing DataLoader")
    print("=" * 60)

    from torch.utils.data import DataLoader
    from utils.train_utils import load_config
    from dataset.ssv2_dataset import build_pretraining_dataset

    config = load_config('configs/videomae_tiny_ssv2.yaml')
    dataset = build_pretraining_dataset(config)

    loader = DataLoader(dataset, batch_size=2, shuffle=True,
                        num_workers=0, drop_last=True)

    print(f"\n  Loading a batch of 2...")
    t0 = time.time()
    videos, masks = next(iter(loader))
    t1 = time.time()
    print(f"  Batch videos: {videos.shape}")
    print(f"  Batch masks:  {masks.shape}")
    print(f"   Batch time:   {t1 - t0:.2f}s")

    return True


def main():
    print("VideoMAE Smoke Test")
    print("=" * 60)

    try:
        ok = test_model()
        if not ok:
            sys.exit(1)
    except Exception as e:
        print(f"  Model test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        ok = test_dataset()
        if not ok:
            sys.exit(1)
    except Exception as e:
        print(f"  Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        ok = test_dataloader()
        if not ok:
            sys.exit(1)
    except Exception as e:
        print(f"  DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Pre-train:  python train_pretrain.py --config configs/videomae_tiny_ssv2.yaml")
    print("  2. Fine-tune:  python train_finetune.py --config configs/videomae_tiny_ssv2.yaml")
    print("  3. Evaluate:   python evaluate.py --config configs/videomae_tiny_ssv2.yaml --checkpoint output/finetune/checkpoint_best.pth")


if __name__ == '__main__':
    main()
