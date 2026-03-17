"""
Core PyTorch Dataset for Sentinel-2 irrigation patches.

Loads GeoTIFF patches and labels, extracts configured bands/seasons,
applies transforms, and returns tensors ready for training.

Key design decisions:
- Reads raw GeoTIFFs with rasterio (not pre-converted to numpy)
- Supports configurable band groups via BandConfig
- Handles ignore_index=255 for masked labels
- Lazy loading (no full dataset in memory)
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
    """

    def __init__(
        self,
        data_root: str | Path,
        tile_ids: list[int],
        band_config: BandConfig,
        transform=None,
        label_transform=None,
        return_metadata: bool = False,
    ):
        self.data_root = Path(data_root)
        self.tile_ids = sorted(tile_ids)
        self.band_config = band_config
        self.transform = transform
        self.label_transform = label_transform
        self.return_metadata = return_metadata

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

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> dict:
        tid = self.tile_ids[idx]
        tile_name = f"tile_{tid:04d}"

        # Load and stack bands across seasons
        all_bands = []
        for season in self.band_config.seasons:
            img_path = self.data_root / "images" / f"{tile_name}_{season}.tif"
            with rasterio.open(img_path) as src:
                # src.read() returns (bands, H, W) as float32
                data = src.read()  # (14, 224, 224)
            selected = data[self.band_config.band_indices]  # (n_bands, 224, 224)
            all_bands.append(selected)

        image = np.concatenate(all_bands, axis=0)  # (num_channels, 224, 224)

        # Load label
        label_path = self.data_root / "labels" / f"{tile_name}_label.tif"
        with rasterio.open(label_path) as src:
            label = src.read(1)  # (224, 224) as uint8

        # Apply label cleaning (erosion, ignore masking) if configured
        if self.label_transform is not None:
            label = self.label_transform(label, image)

        # Apply spatial augmentations (albumentations expects HWC for image)
        if self.transform is not None:
            # albumentations needs (H, W, C) for image
            image_hwc = np.transpose(image, (1, 2, 0))  # (224, 224, C)
            transformed = self.transform(image=image_hwc, mask=label)
            image_hwc = transformed["image"]
            label = transformed["mask"]
            image = np.transpose(image_hwc, (2, 0, 1))  # (C, 224, 224)

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
