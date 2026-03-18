"""
Core PyTorch Dataset for Sentinel-2 irrigation patches.

Loads GeoTIFF patches and labels, extracts configured bands/seasons,
applies transforms, and returns tensors ready for training.

Key design decisions:
- Optional numpy caching: converts GeoTIFFs to .npy on first access
  for much faster subsequent reads (avoids rasterio overhead per iteration)
- Supports configurable band groups via BandConfig
- Handles ignore_index=255 for masked labels
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio
import numpy as np

from irrigation.data.bands import BandConfig


class IrrigationDataset(Dataset):
    """
    Sentinel-2 irrigation patch dataset.

    Args:
        data_root: Path to state directory (e.g., /shared/data/irrigation/Colorado)
        tile_ids: List of tile IDs to include (for split management)
        band_config: BandConfig defining which channels to load
        transform: Optional albumentations transform pipeline
        label_transform: Optional label cleaning function (erosion, ignore masking)
        return_metadata: Whether to include metadata dict in output
        use_cache: If True, cache tiles as .npy files for faster reads
        normalize_mean: Per-channel mean for z-score normalization (shape: num_channels)
        normalize_std: Per-channel std for z-score normalization (shape: num_channels)
    """

    def __init__(
        self,
        data_root: str | Path,
        tile_ids: list[int],
        band_config: BandConfig,
        transform=None,
        label_transform=None,
        return_metadata: bool = False,
        use_cache: bool = True,
        normalize_mean: np.ndarray | None = None,
        normalize_std: np.ndarray | None = None,
    ):
        self.data_root = Path(data_root)
        self.tile_ids = sorted(tile_ids)
        self.band_config = band_config
        self.transform = transform
        self.label_transform = label_transform
        self.return_metadata = return_metadata
        self.use_cache = use_cache
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        # Cache directory sits alongside images/labels
        self.cache_dir = self.data_root / "npy_cache"

        # Validate that all tile files exist
        self._validate_tiles()

    def _validate_tiles(self):
        """Check that all expected files exist for all tiles and seasons."""
        missing = []
        for tid in self.tile_ids:
            tile_name = f"tile_{tid:04d}"
            label_path = self.data_root / "labels" / f"{tile_name}_label.tif"
            if not label_path.exists():
                missing.append(str(label_path))
            for season in self.band_config.seasons:
                img_path = self.data_root / "images" / f"{tile_name}_{season}.tif"
                if not img_path.exists():
                    missing.append(str(img_path))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} files. First 5: {missing[:5]}"
            )

    def _load_image_tif(self, tile_name: str, season: str) -> np.ndarray:
        """Load image from GeoTIFF."""
        img_path = self.data_root / "images" / f"{tile_name}_{season}.tif"
        with rasterio.open(img_path) as src:
            return src.read()  # (14, 224, 224)

    def _load_label_tif(self, tile_name: str) -> np.ndarray:
        """Load label from GeoTIFF."""
        label_path = self.data_root / "labels" / f"{tile_name}_label.tif"
        with rasterio.open(label_path) as src:
            return src.read(1)  # (224, 224)

    def _get_cache_path(self, tile_name: str, kind: str) -> Path:
        """Get numpy cache file path for a tile."""
        return self.cache_dir / f"{tile_name}_{kind}.npy"

    def _load_image_cached(self, tile_name: str, season: str) -> np.ndarray:
        """Load image, using numpy cache if available."""
        cache_path = self._get_cache_path(tile_name, f"img_{season}")
        if cache_path.exists():
            return np.load(cache_path)
        data = self._load_image_tif(tile_name, season)
        # Write cache (create dir if needed)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, data)
        return data

    def _load_label_cached(self, tile_name: str) -> np.ndarray:
        """Load label, using numpy cache if available."""
        cache_path = self._get_cache_path(tile_name, "label")
        if cache_path.exists():
            return np.load(cache_path)
        data = self._load_label_tif(tile_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, data)
        return data

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> dict:
        tid = self.tile_ids[idx]
        tile_name = f"tile_{tid:04d}"

        # Load and stack bands across seasons
        all_bands = []
        full_bands = []  # full 14-band data for label transform (e.g. NDVI noise masking)
        for season in self.band_config.seasons:
            if self.use_cache:
                data = self._load_image_cached(tile_name, season)
            else:
                data = self._load_image_tif(tile_name, season)
            selected = data[self.band_config.band_indices]  # (n_bands, 224, 224)
            all_bands.append(selected)
            if self.label_transform is not None:
                full_bands.append(data)

        image = np.concatenate(all_bands, axis=0)  # (num_channels, 224, 224)

        # Load label
        if self.use_cache:
            label = self._load_label_cached(tile_name)
        else:
            label = self._load_label_tif(tile_name)

        # Apply label cleaning (erosion, ignore masking) if configured
        # Use full-band image so noise functions can access any band (e.g. NDVI at index 9)
        if self.label_transform is not None:
            full_image = np.concatenate(full_bands, axis=0)
            label = self.label_transform(label, full_image)

        # Apply spatial augmentations (albumentations expects HWC for image)
        if self.transform is not None:
            # albumentations needs (H, W, C) for image
            image_hwc = np.transpose(image, (1, 2, 0))  # (224, 224, C)
            transformed = self.transform(image=image_hwc, mask=label)
            image_hwc = transformed["image"]
            label = transformed["mask"]
            image = np.transpose(image_hwc, (2, 0, 1))  # (C, 224, 224)

        # Per-channel z-score normalization
        if self.normalize_mean is not None and self.normalize_std is not None:
            image = image.astype(np.float32)
            # image is (C, H, W) — normalize each channel
            mean = self.normalize_mean[:, None, None]  # (C, 1, 1)
            std = self.normalize_std[:, None, None]  # (C, 1, 1)
            image = (image - mean) / std

        # Convert to tensors
        image_tensor = torch.from_numpy(image.copy()).float()
        label_tensor = torch.from_numpy(label.copy()).long()

        output = {
            "image": image_tensor,  # (C, 224, 224)
            "label": label_tensor,  # (224, 224)
        }

        if self.return_metadata:
            output["metadata"] = {
                "tile_id": tid,
                "tile_name": tile_name,
                "state": self.data_root.name,
            }

        return output
