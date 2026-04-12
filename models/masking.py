"""
Tube Masking Generator for VideoMAE pre-training.
Reference: https://github.com/MCG-NJU/VideoMAE/blob/main/masking_generator.py

Tube masking applies the same spatial mask across all temporal positions,
ensuring temporal consistency in the masking pattern.
"""
import numpy as np


class TubeMaskingGenerator:
    """
    Generate tube masking for video patches.
    
    The same spatial mask pattern is shared across all temporal positions.
    This encourages the model to learn temporal dynamics rather than
    simply copying from adjacent frames.
    
    Args:
        input_size: tuple of (T_patches, H_patches, W_patches)
        mask_ratio: fraction of patches to mask (e.g., 0.9 for 90%)
    """
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        return (f"TubeMasking(total={self.total_patches}, "
                f"masked={self.total_masks}, "
                f"visible={self.total_patches - self.total_masks})")

    def __call__(self):
        """
        Returns:
            mask: np.ndarray of shape [total_patches], 
                  1 = masked, 0 = visible
        """
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        # Tile the same spatial mask across all temporal positions
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask


class FlowGuidedPersistentMaskingGenerator:
    """
    Optical-flow-guided persistent patch masking for VideoMAE pre-training.

    At the reference temporal step (t=0) a random set of spatial patches is
    selected for masking.  Dense optical flow (Farneback, via OpenCV) is then
    used to track the pixel-space centres of those patches through every
    subsequent temporal step.  The tracked positions are mapped back to
    patch-grid indices to produce each step's spatial mask.

    To guarantee pipeline compatibility the masked-patch count per temporal
    step is kept exactly equal to ``int(mask_ratio * patches_per_frame)``
    by replenishing (if tracked patches converge) or pruning (if they
    diverge beyond the target count).

    Args:
        input_size:   (T_patches, H_patches, W_patches)
        mask_ratio:   fraction of patches to mask per frame (e.g. 0.9)
        patch_size:   spatial patch size in pixels
        tubelet_size: temporal patch size in frames
    """

    def __init__(self, input_size, mask_ratio, patch_size, tubelet_size):
        self.frames_t, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames_t * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames_t * self.num_masks_per_frame
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

    def __repr__(self):
        return (
            f"FlowGuidedPersistentMasking(total={self.total_patches}, "
            f"masked={self.total_masks}, "
            f"visible={self.total_patches - self.total_masks})"
        )

    def __call__(self, gray_frames):
        """
        Generate a flow-guided persistent mask from grayscale video frames.

        Args:
            gray_frames: np.ndarray [T, H, W] uint8 grayscale,
                         where T = num_frames (raw temporal length before
                         tubelet patchification).
        Returns:
            mask: np.ndarray [total_patches], 1 = masked, 0 = visible
        """
        import cv2  # local import so tube-only configs never need cv2 here

        T, H, W = gray_frames.shape
        ps = self.patch_size

        # Representative frame index for each temporal patch step
        ref_indices = [t * self.tubelet_size for t in range(self.frames_t)]

        # ── randomly select spatial patches to mask at t=0 ────────
        all_idx = np.arange(self.num_patches_per_frame)
        np.random.shuffle(all_idx)
        masked_ref = all_idx[: self.num_masks_per_frame]

        # Convert patch indices to pixel centres
        ref_rows = masked_ref // self.width
        ref_cols = masked_ref % self.width
        centres_x = (ref_cols * ps + ps / 2.0).astype(np.float32)
        centres_y = (ref_rows * ps + ps / 2.0).astype(np.float32)
        current_pos = np.stack([centres_x, centres_y], axis=1)  # [N_mask, 2]

        # ── track centres through time with Farneback flow ────────
        all_positions = [current_pos.copy()]

        for t in range(1, self.frames_t):
            prev_gray = gray_frames[ref_indices[t - 1]]
            curr_gray = gray_frames[ref_indices[t]]

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                flow=None, pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            # Sample flow at current tracked positions (vectorised)
            prev_pos = all_positions[-1]
            ix = np.clip(np.round(prev_pos[:, 0]).astype(int), 0, W - 1)
            iy = np.clip(np.round(prev_pos[:, 1]).astype(int), 0, H - 1)
            displacements = flow[iy, ix]  # [N_mask, 2] (dx, dy)
            all_positions.append(prev_pos + displacements)

        # ── assemble full mask ────────────────────────────────────
        mask = np.zeros(self.total_patches)

        for t in range(self.frames_t):
            pos = all_positions[t]

            # Map pixel positions → patch grid indices
            pcol = np.clip(
                np.floor(pos[:, 0] / ps).astype(int), 0, self.width - 1
            )
            prow = np.clip(
                np.floor(pos[:, 1] / ps).astype(int), 0, self.height - 1
            )
            tracked = np.unique(prow * self.width + pcol)

            spatial = np.zeros(self.num_patches_per_frame)
            spatial[tracked] = 1.0

            # Enforce exact mask count for pipeline compatibility
            count = int(spatial.sum())
            if count < self.num_masks_per_frame:
                # Patches converged — replenish with random unmasked patches
                unmasked = np.where(spatial == 0)[0]
                np.random.shuffle(unmasked)
                spatial[unmasked[: self.num_masks_per_frame - count]] = 1.0
            elif count > self.num_masks_per_frame:
                masked_arr = np.where(spatial == 1)[0]
                np.random.shuffle(masked_arr)
                spatial[masked_arr[self.num_masks_per_frame :]] = 0.0

            start = t * self.num_patches_per_frame
            mask[start : start + self.num_patches_per_frame] = spatial

        return mask
