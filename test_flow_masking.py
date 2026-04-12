"""
Quick integration test for flow-guided persistent masking.
Validates mask shape, dtype, count, and model forward compatibility.
"""
import sys
import numpy as np
import torch

# ── 1. Unit test: FlowGuidedPersistentMaskingGenerator ─────────────────
print("=" * 60)
print("1. Testing FlowGuidedPersistentMaskingGenerator")
print("=" * 60)

from models.masking import FlowGuidedPersistentMaskingGenerator, TubeMaskingGenerator

# Config: 8 frames, tubelet=2 → 4 temporal steps; 112px, patch=16 → 7x7 grid
gen = FlowGuidedPersistentMaskingGenerator(
    input_size=(4, 7, 7), mask_ratio=0.9, patch_size=16, tubelet_size=2
)
print(f"  Generator: {gen}")

# Synthetic grayscale video [T=8, H=112, W=112]
np.random.seed(42)
gray_frames = np.random.randint(0, 256, (8, 112, 112), dtype=np.uint8)

mask = gen(gray_frames)
print(f"  Mask shape: {mask.shape} (expect (196,))")
print(f"  Mask dtype: {mask.dtype}")
print(f"  Masked patches: {int(mask.sum())} (expect 176)")
print(f"  Visible patches: {int((mask == 0).sum())} (expect 20)")
assert mask.shape == (196,), f"Shape mismatch: {mask.shape}"
assert int(mask.sum()) == 176, f"Mask count mismatch: {int(mask.sum())}"
print("  PASSED")

# Compare with tube masking
tube_gen = TubeMaskingGenerator(input_size=(4, 7, 7), mask_ratio=0.9)
tube_mask = tube_gen()
assert tube_mask.shape == mask.shape, "Shape mismatch vs tube"
assert int(tube_mask.sum()) == int(mask.sum()), "Count mismatch vs tube"
print("  Tube masking cross-check: PASSED")

# ── 2. Test per-frame mask count consistency ───────────────────────────
print("\n" + "=" * 60)
print("2. Per-frame mask count consistency")
print("=" * 60)

for t in range(4):
    frame_mask = mask[t * 49 : (t + 1) * 49]
    count = int(frame_mask.sum())
    print(f"  Frame {t}: masked={count} visible={49 - count}")
    assert count == 44, f"Frame {t} count {count} != 44"
print("  All frames have exactly 44/49 masked: PASSED")

# ── 3. Dataset integration test ────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Dataset integration (flow_guided_persistent)")
print("=" * 60)

from utils.train_utils import load_config
from dataset.ssv2_dataset import build_pretraining_dataset

config = load_config('configs/videomae_tiny_ssv2_flow.yaml')
assert config['masking'].get('mode') == 'flow_guided_persistent', \
    f"Expected flow_guided_persistent, got {config['masking'].get('mode')}"

ds = build_pretraining_dataset(config)
print(f"  Dataset size: {len(ds)}")
print(f"  Masking mode: {ds.masking_mode}")

# Load one sample
import time
t0 = time.time()
video, mask = ds[0]
t1 = time.time()

print(f"  Video shape: {video.shape} dtype: {video.dtype}")
print(f"  Mask shape:  {mask.shape}  dtype: {mask.dtype}")
print(f"  Mask True count:  {mask.sum().item()} (expect 176)")
print(f"  Mask False count: {(~mask).sum().item()} (expect 20)")
print(f"  Load time: {t1-t0:.2f}s")

assert video.shape == torch.Size([3, 8, 112, 112]), f"Video shape: {video.shape}"
assert mask.shape == torch.Size([196]), f"Mask shape: {mask.shape}"
assert mask.dtype == torch.bool, f"Mask dtype: {mask.dtype}"
assert mask.sum().item() == 176, f"Mask count: {mask.sum().item()}"
print("  PASSED")

# ── 4. Model forward pass compatibility ────────────────────────────────
print("\n" + "=" * 60)
print("4. Model forward pass with flow-guided mask")
print("=" * 60)

from models.videomae import build_pretrain_model

model = build_pretrain_model(config)
B = 2
dummy_video = torch.randn(B, 3, 8, 112, 112)
dummy_mask = mask.unsqueeze(0).expand(B, -1)  # [B, 196]

with torch.no_grad():
    pred = model(dummy_video, dummy_mask)

print(f"  Model output: {pred.shape} (expect [{B}, 176, 1536])")
assert pred.shape == torch.Size([B, 176, 1536]), f"Output shape: {pred.shape}"
print("  PASSED")

# ── 5. Backward compat: tube mode still works ─────────────────────────
print("\n" + "=" * 60)
print("5. Backward compatibility: tube mode")
print("=" * 60)

config_tube = load_config('configs/videomae_tiny_ssv2_static.yaml')
config_tube['masking']['mode'] = 'tube'
ds_tube = build_pretraining_dataset(config_tube)
print(f"  Tube dataset masking_mode: {ds_tube.masking_mode}")
video_t, mask_t = ds_tube[0]
print(f"  Video shape: {video_t.shape}")
print(f"  Mask shape:  {mask_t.shape} count: {mask_t.sum().item()}")
assert mask_t.sum().item() == 176
print("  PASSED")

# ── 6. Backward compat: missing mode key defaults to tube ─────────────
print("\n" + "=" * 60)
print("6. Backward compatibility: missing mode key")
print("=" * 60)

config_nomode = load_config('configs/videomae_tiny_ssv2_flow.yaml')
del config_nomode['masking']['mode']
ds_nomode = build_pretraining_dataset(config_nomode)
print(f"  Default masking_mode: {ds_nomode.masking_mode}")
assert ds_nomode.masking_mode == 'tube'
print("  PASSED")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
