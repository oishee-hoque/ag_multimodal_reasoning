"""Tests for the IrrigationDataset."""

import numpy as np
import pytest
import torch

from irrigation.data.dataset import IrrigationDataset
from irrigation.data.bands import get_band_config
from irrigation.data.transforms import get_train_transforms, get_val_transforms


class TestIrrigationDataset:
    """Test dataset loading and tensor shapes."""

    def test_rgb_dataset_shapes(self, tmp_data_dir, rgb_band_config):
        """RGB dataset returns (3, 224, 224) images and (224, 224) labels."""
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=list(range(5)),
            band_config=rgb_band_config,
        )
        assert len(ds) == 5
        sample = ds[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["label"].shape == (224, 224)
        assert sample["image"].dtype == torch.float32
        assert sample["label"].dtype == torch.int64

    def test_spectral_dataset_shapes(self, tmp_data_dir, spectral_band_config):
        """Spectral dataset returns (14, 224, 224) images."""
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=list(range(3)),
            band_config=spectral_band_config,
        )
        sample = ds[0]
        assert sample["image"].shape == (14, 224, 224)

    def test_temporal_dataset_shapes(self, tmp_data_dir, temporal_band_config):
        """Temporal dataset returns (42, 224, 224) images."""
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=list(range(3)),
            band_config=temporal_band_config,
        )
        sample = ds[0]
        assert sample["image"].shape == (42, 224, 224)

    def test_metadata_returned(self, tmp_data_dir, rgb_band_config):
        """Dataset returns metadata when requested."""
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0, 1],
            band_config=rgb_band_config,
            return_metadata=True,
        )
        sample = ds[0]
        assert "metadata" in sample
        assert sample["metadata"]["tile_id"] == 0
        assert sample["metadata"]["tile_name"] == "tile_0000"

    def test_missing_files_raises(self, tmp_data_dir, rgb_band_config):
        """Dataset raises FileNotFoundError for missing tiles."""
        with pytest.raises(FileNotFoundError, match="Missing"):
            IrrigationDataset(
                data_root=tmp_data_dir,
                tile_ids=[999],
                band_config=rgb_band_config,
            )

    def test_ignore_index_preserved(self, tmp_data_dir, rgb_band_config):
        """Labels with ignore_index=255 are preserved as-is."""
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0],
            band_config=rgb_band_config,
        )
        sample = ds[0]
        label = sample["label"].numpy()
        # Our fixture sets [0:10, 0:10] to 255
        assert (label[0:10, 0:10] == 255).all()

    def test_transforms_applied(self, tmp_data_dir, rgb_band_config):
        """Transforms are applied without errors."""
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0, 1],
            band_config=rgb_band_config,
            transform=get_train_transforms(),
        )
        sample = ds[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["label"].shape == (224, 224)

    def test_sorted_tile_ids(self, tmp_data_dir, rgb_band_config):
        """Tile IDs are sorted regardless of input order."""
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[3, 1, 2],
            band_config=rgb_band_config,
        )
        assert ds.tile_ids == [1, 2, 3]


