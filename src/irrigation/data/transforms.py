"""
Augmentation pipelines using albumentations.

Design choices:
- Only geometric augmentations (flips, rotations, transpose)
- NO color/spectral augmentations — the spectral values are physically
  meaningful (reflectance, indices) and shouldn't be distorted
- All transforms are applied identically to image and mask
"""

import albumentations as A


def get_train_transforms() -> A.Compose:
    """Training augmentations: geometric only."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
    ])


def get_val_transforms() -> A.Compose:
    """Validation/test: no augmentation."""
    return A.Compose([])
