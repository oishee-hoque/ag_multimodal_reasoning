"""Tests for the IrrigationDataset."""

import numpy as np
import pytest
import torch

from irrigation.data.dataset import IrrigationDataset
from irrigation.data.bands import get_band_config
from irrigation.data.transforms import get_train_transforms, get_val_transforms
from irrigation.data.noise import erode_labels, ndvi_ignore_mask, combined_cleaning


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

    def test_label_transform_applied(self, tmp_data_dir, spectral_band_config):
        """Label transform (erosion) modifies labels."""
        ds_clean = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0],
            band_config=spectral_band_config,
        )
        ds_eroded = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0],
            band_config=spectral_band_config,
            label_transform=erode_labels,
        )

        label_clean = ds_clean[0]["label"].numpy()
        label_eroded = ds_eroded[0]["label"].numpy()

        # Erosion should increase the number of ignore (255) pixels
        assert (label_eroded == 255).sum() >= (label_clean == 255).sum()

    def test_sorted_tile_ids(self, tmp_data_dir, rgb_band_config):
        """Tile IDs are sorted regardless of input order."""
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[3, 1, 2],
            band_config=rgb_band_config,
        )
        assert ds.tile_ids == [1, 2, 3]


class TestNoiseReduction:
    """Test label cleaning functions."""

    def test_erode_labels_basic(self):
        """Erosion shrinks labeled regions and sets edges to 255."""
        label = np.zeros((50, 50), dtype=np.uint8)
        label[10:40, 10:40] = 1  # 30x30 block of class 1
        image = np.zeros((3, 50, 50), dtype=np.float32)

        cleaned = erode_labels(label, image, erosion_pixels=2)

        # Original class 1 area
        original_count = (label == 1).sum()
        # Eroded class 1 area should be smaller
        eroded_count = (cleaned == 1).sum()
        # Eroded-away pixels should be 255
        ignore_count = (cleaned == 255).sum()

        assert eroded_count < original_count
        assert ignore_count > 0
        assert eroded_count + ignore_count == original_count

    def test_erode_labels_no_background_erosion(self):
        """Background (0) pixels are never eroded."""
        label = np.zeros((50, 50), dtype=np.uint8)
        image = np.zeros((3, 50, 50), dtype=np.float32)

        cleaned = erode_labels(label, image, erosion_pixels=2)
        np.testing.assert_array_equal(label, cleaned)

    def test_ndvi_ignore_mask(self):
        """High-NDVI background pixels become ignore (255)."""
        label = np.zeros((50, 50), dtype=np.uint8)
        # 14-channel image, NDVI is band 9
        image = np.zeros((14, 50, 50), dtype=np.float32)
        image[9, 20:30, 20:30] = 0.6  # High NDVI patch

        cleaned = ndvi_ignore_mask(
            label, image, ndvi_band_index=9, ndvi_threshold=0.4, seasons_axis_size=1
        )

        # High NDVI area that was background should now be 255
        assert (cleaned[20:30, 20:30] == 255).all()
        # Rest should still be 0
        assert (cleaned[0:20, 0:20] == 0).all()

    def test_ndvi_mask_preserves_labeled(self):
        """NDVI masking only affects background (0) pixels."""
        label = np.ones((50, 50), dtype=np.uint8)  # all class 1
        image = np.zeros((14, 50, 50), dtype=np.float32)
        image[9] = 0.8  # High NDVI everywhere

        cleaned = ndvi_ignore_mask(label, image)
        np.testing.assert_array_equal(label, cleaned)

    def test_combined_cleaning(self):
        """Combined cleaning applies both erosion and NDVI masking."""
        label = np.zeros((50, 50), dtype=np.uint8)
        label[10:40, 10:40] = 2
        image = np.zeros((14, 50, 50), dtype=np.float32)
        image[9, 0:10, 0:10] = 0.6  # High NDVI in background

        cleaned = combined_cleaning(
            label, image, erosion_pixels=2, ndvi_band_index=9,
            ndvi_threshold=0.4, seasons_axis_size=1,
        )

        # Both erosion and NDVI masking should have produced 255 pixels
        assert (cleaned == 255).sum() > 0
        # High NDVI background should be 255
        assert (cleaned[0:10, 0:10] == 255).all()
