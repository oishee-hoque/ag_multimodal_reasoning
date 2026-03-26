"""
Core PyTorch Dataset for Sentinel-2 irrigation patches.

Loads GeoTIFF patches and labels, extracts configured bands/seasons,
applies transforms, and returns tensors ready for training.

Key design decisions:
- Optional numpy caching: converts GeoTIFFs to .npy on first access
  for much faster subsequent reads (avoids rasterio overhead per iteration)
- Supports configurable band groups via BandConfig
- Handles ignore_index=255 for masked labels
- Raw pixel values are used directly (no per-band normalization)
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio
import numpy as np

from irrigation.data.bands import BandConfig
from irrigation.data.field_channels import create_all_field_channels


class IrrigationDataset(Dataset):
    """
    Sentinel-2 irrigation patch dataset.

    Args:
        data_root: Path to state directory (e.g., /shared/data/irrigation/Colorado)
        tile_ids: List of tile IDs to include (for split management)
        band_config: BandConfig defining which channels to load
        transform: Optional albumentations transform pipeline
        return_metadata: Whether to include metadata dict in output
        use_cache: If True, cache tiles as .npy files for faster reads
        noise_strategy: Label noise refinement strategy. Options:
            - None: no refinement
            - "ndvi_bidirectional": suppress noisy labels using NDVI thresholds
            - "ndvi_background_only": only suppress background pixels with high NDVI
        ndvi_high_threshold: NDVI above this for background pixels → set to ignore (likely unlabeled irrigated)
        ndvi_low_threshold: NDVI below this for irrigated pixels → set to ignore (likely mislabeled)
        ndvi_band_index: Index of NDVI_median band in the 14-band GeoTIFF (default 9)
        ndvi_season: Season to use for NDVI check (default "s4", peak summer)
    """

    def __init__(
        self,
        data_root: str | Path,
        tile_ids: list[int],
        band_config: BandConfig,
        transform=None,
        return_metadata: bool = False,
        use_cache: bool = True,
        noise_strategy: str | None = None,
        ndvi_high_threshold: float = 0.4,
        ndvi_low_threshold: float = 0.15,
        ndvi_band_index: int = 9,
        ndvi_season: str = "s4",
        use_field_channels: bool = False,
    ):
        self.data_root = Path(data_root)
        self.tile_ids = sorted(tile_ids)
        self.band_config = band_config
        self.transform = transform
        self.return_metadata = return_metadata
        self.use_cache = use_cache
        self.noise_strategy = noise_strategy
        self.ndvi_high_threshold = ndvi_high_threshold
        self.ndvi_low_threshold = ndvi_low_threshold
        self.ndvi_band_index = ndvi_band_index
        self.ndvi_season = ndvi_season
        self.use_field_channels = use_field_channels

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

    def _load_ndvi(self, tile_name: str) -> np.ndarray:
        """Load the NDVI band from the reference season."""
        if self.use_cache:
            img_data = self._load_image_cached(tile_name, self.ndvi_season)
        else:
            img_data = self._load_image_tif(tile_name, self.ndvi_season)
        return img_data[self.ndvi_band_index]  # (224, 224)

    def _refine_labels(self, label: np.ndarray, tile_name: str) -> np.ndarray:
        """Apply NDVI-based bidirectional label noise suppression.

        - Irrigated pixels (class 1,2,3) with NDVI < low_threshold → ignore (255)
          These are likely mislabeled or fallow fields.
        - Background pixels (class 0) with NDVI > high_threshold → ignore (255)
          These are likely unlabeled irrigated fields.
        """
        ndvi = self._load_ndvi(tile_name)
        label = label.copy()

        # Suppress irrigated pixels with low NDVI (likely mislabeled)
        irrigated_mask = (label >= 1) & (label <= 3)
        low_ndvi = ndvi < self.ndvi_low_threshold
        label[irrigated_mask & low_ndvi] = 255

        # Suppress background pixels with high NDVI (likely unlabeled irrigated)
        bg_mask = label == 0
        high_ndvi = ndvi > self.ndvi_high_threshold
        label[bg_mask & high_ndvi] = 255

        return label

    def _refine_labels_background_only(self, label: np.ndarray, tile_name: str) -> np.ndarray:
        """Suppress only background pixels with high NDVI (one-directional).

        Background pixels (class 0) with NDVI > high_threshold → ignore (255).
        Irrigated pixels are never modified regardless of their NDVI.
        """
        ndvi = self._load_ndvi(tile_name)
        label = label.copy()

        bg_mask = label == 0
        high_ndvi = ndvi > self.ndvi_high_threshold
        label[bg_mask & high_ndvi] = 255

        return label

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> dict:
        tid = self.tile_ids[idx]
        tile_name = f"tile_{tid:04d}"

        # Load and stack bands across seasons
        all_bands = []
        for season in self.band_config.seasons:
            if self.use_cache:
                data = self._load_image_cached(tile_name, season)
            else:
                data = self._load_image_tif(tile_name, season)
            selected = data[self.band_config.band_indices]  # (n_bands, 224, 224)
            all_bands.append(selected)

        image = np.concatenate(all_bands, axis=0)  # (num_channels, 224, 224)

        # Load label
        if self.use_cache:
            label = self._load_label_cached(tile_name)
        else:
            label = self._load_label_tif(tile_name)

        # Generate field channels from ORIGINAL label (before noise cleaning)
        if self.use_field_channels:
            field_channels = create_all_field_channels(label)  # (3, H, W)

        # Apply label noise refinement
        if self.noise_strategy == "ndvi_bidirectional":
            label = self._refine_labels(label, tile_name)
        elif self.noise_strategy == "ndvi_background_only":
            label = self._refine_labels_background_only(label, tile_name)

        # Apply spatial augmentations (albumentations expects HWC for image)
        if self.transform is not None:
            image_hwc = np.transpose(image, (1, 2, 0))  # (224, 224, C)
            if self.use_field_channels:
                # Include field channels in augmentation so they get the same
                # spatial transforms (flips, rotations) as the image
                field_hwc = np.transpose(field_channels, (1, 2, 0))
                combined_hwc = np.concatenate([image_hwc, field_hwc], axis=2)
                transformed = self.transform(image=combined_hwc, mask=label)
                combined_hwc = transformed["image"]
                label = transformed["mask"]
                n_img_channels = image_hwc.shape[2]
                image = np.transpose(combined_hwc[:, :, :n_img_channels], (2, 0, 1))
                field_channels = np.transpose(combined_hwc[:, :, n_img_channels:], (2, 0, 1))
            else:
                transformed = self.transform(image=image_hwc, mask=label)
                image_hwc = transformed["image"]
                label = transformed["mask"]
                image = np.transpose(image_hwc, (2, 0, 1))  # (C, 224, 224)

        # Concatenate field channels to image
        if self.use_field_channels:
            image = np.concatenate([image, field_channels], axis=0)  # (C+3, H, W)

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
