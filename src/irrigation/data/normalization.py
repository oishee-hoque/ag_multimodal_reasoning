"""
Per-band normalization for Sentinel-2 imagery.

Best practice for satellite imagery training:
- Z-score normalization (subtract mean, divide by std) per channel
- Statistics computed from training set only (no data leakage)
- Same stats applied to val/test
- For RGB with ImageNet pretrained encoders: use ImageNet statistics
  so pretrained features transfer properly (requires scaling to [0,1] first)
"""

import numpy as np
import rasterio
from pathlib import Path

from irrigation.data.bands import BandConfig

# ImageNet statistics — assumes input scaled to [0, 1]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Sentinel-2 L2A reflectance scaling factor (DN / 10000 → [0,1] reflectance)
S2_SCALE_FACTOR = 10000.0


def compute_band_statistics(
    data_root: Path,
    tile_ids: list[int],
    band_config: BandConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from training tiles.

    Iterates all training tiles, accumulating sums for a two-pass
    calculation without loading everything into memory at once.

    Args:
        data_root: Path to state directory
        tile_ids: Training tile IDs only (no val/test — prevents leakage)
        band_config: Defines which bands/seasons to load

    Returns:
        (mean, std) arrays of shape (num_channels,)
    """
    num_channels = band_config.num_channels
    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_sq_sum = np.zeros(num_channels, dtype=np.float64)
    pixel_count = 0

    for tid in tile_ids:
        tile_name = f"tile_{tid:04d}"
        all_bands = []
        for season in band_config.seasons:
            img_path = data_root / "images" / f"{tile_name}_{season}.tif"
            with rasterio.open(img_path) as src:
                data = src.read().astype(np.float64)  # (14, H, W)
            selected = data[band_config.band_indices]
            all_bands.append(selected)

        image = np.concatenate(all_bands, axis=0)  # (num_channels, H, W)
        n_pixels = image.shape[1] * image.shape[2]

        channel_sum += image.sum(axis=(1, 2))
        channel_sq_sum += (image**2).sum(axis=(1, 2))
        pixel_count += n_pixels

    mean = (channel_sum / pixel_count).astype(np.float32)
    variance = (channel_sq_sum / pixel_count) - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(variance, 0)).astype(np.float32)

    # Prevent division by zero for constant bands
    std = np.maximum(std, 1e-6)

    return mean, std


def get_imagenet_stats() -> tuple[np.ndarray, np.ndarray]:
    """Get ImageNet mean/std for 3-channel RGB input scaled to [0, 1]."""
    return IMAGENET_MEAN.copy(), IMAGENET_STD.copy()
