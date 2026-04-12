"""
Video transforms for VideoMAE training.
Handles temporal sampling, spatial augmentation, and normalization
for video tensors of shape [T, H, W, C] (from decord) → [C, T, H, W] (PyTorch).
"""
import torch
import numpy as np
import cv2
from torchvision import transforms


class VideoTransform:
    """
    Video transform pipeline for training/validation.
    Input: list of numpy frames [T x H x W x C] (uint8, BGR from cv2 / RGB from decord)
    Output: torch.Tensor [C, T, H, W] (float32, normalized)
    """
    def __init__(self, img_size=128, mode='train'):
        self.img_size = img_size
        self.mode = mode
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, frames):
        """
        Args:
            frames: numpy array of shape [T, H, W, C] (uint8, RGB)
        Returns:
            tensor: [C, T, H, W] (float32, normalized)
        """
        T, H, W, C = frames.shape

        if self.mode == 'train':
            frames = self._random_resized_crop(frames)
            frames = self._random_horizontal_flip(frames)
        else:
            frames = self._center_crop_resize(frames)

        # Normalize: [0, 255] → [0, 1] → normalized
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - self.mean) / self.std

        # [T, H, W, C] → [C, T, H, W]
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
        return tensor

    def _random_resized_crop(self, frames):
        """Random resized crop for training."""
        T, H, W, C = frames.shape
        # Random scale between 0.5 and 1.0
        scale = np.random.uniform(0.5, 1.0)
        crop_h = int(H * scale)
        crop_w = int(W * scale)

        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)

        frames = frames[:, top:top+crop_h, left:left+crop_w, :]

        # Resize to target size
        resized = np.zeros((T, self.img_size, self.img_size, C), dtype=np.uint8)
        for i in range(T):
            resized[i] = cv2.resize(frames[i], (self.img_size, self.img_size),
                                     interpolation=cv2.INTER_LINEAR)
        return resized

    def _random_horizontal_flip(self, frames, p=0.5):
        """Random horizontal flip for training."""
        if np.random.random() < p:
            frames = frames[:, :, ::-1, :].copy()
        return frames

    def _center_crop_resize(self, frames):
        """Center crop and resize for validation."""
        T, H, W, C = frames.shape
        # Center crop to square
        size = min(H, W)
        top = (H - size) // 2
        left = (W - size) // 2
        frames = frames[:, top:top+size, left:left+size, :]

        # Resize to target size
        resized = np.zeros((T, self.img_size, self.img_size, C), dtype=np.uint8)
        for i in range(T):
            resized[i] = cv2.resize(frames[i], (self.img_size, self.img_size),
                                     interpolation=cv2.INTER_LINEAR)
        return resized


def uniform_temporal_subsample(total_frames, num_samples):
    """
    Uniformly sample `num_samples` frame indices from a video of `total_frames`.
    """
    if total_frames <= num_samples:
        # If video is shorter than desired, repeat last frame
        indices = list(range(total_frames))
        while len(indices) < num_samples:
            indices.append(total_frames - 1)
        return indices

    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
    return indices
