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


class TestNoiseRefinement:
    """Test NDVI-based bidirectional label noise suppression."""

    def _make_controlled_tile(self, tmp_data_dir):
        """Overwrite tile_0000 with controlled NDVI and label values."""
        import rasterio
        from rasterio.transform import from_bounds

        h, w = 224, 224
        transform = from_bounds(0, 0, 1, 1, w, h)

        # Create label: top half = irrigated (class 1), bottom half = background (class 0)
        label = np.zeros((h, w), dtype=np.uint8)
        label[:112, :] = 1  # irrigated (flood)

        label_path = tmp_data_dir / "labels" / "tile_0000_label.tif"
        with rasterio.open(
            label_path, "w", driver="GTiff", height=h, width=w,
            count=1, dtype="uint8", transform=transform, crs="EPSG:4326",
        ) as dst:
            dst.write(label, 1)

        # Create image with controlled NDVI (band 9)
        for season in ["s3", "s4", "s5"]:
            img_data = np.zeros((14, h, w), dtype=np.float32)
            ndvi = np.full((h, w), 0.3, dtype=np.float32)  # default: moderate NDVI

            # Top-left quadrant: irrigated pixels with very low NDVI (should be suppressed)
            ndvi[:112, :112] = 0.05

            # Bottom-right quadrant: background pixels with very high NDVI (should be suppressed)
            ndvi[112:, 112:] = 0.6

            img_data[9] = ndvi
            img_path = tmp_data_dir / "images" / f"tile_0000_{season}.tif"
            with rasterio.open(
                img_path, "w", driver="GTiff", height=h, width=w,
                count=14, dtype="float32", transform=transform, crs="EPSG:4326",
            ) as dst:
                dst.write(img_data)

    def test_no_refinement_by_default(self, tmp_data_dir, rgb_band_config):
        """Without noise_strategy, labels are unchanged."""
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0],
            band_config=rgb_band_config,
            use_cache=False,
        )
        sample = ds[0]
        label = sample["label"].numpy()
        # Should contain classes 0-3 and 255, no extra suppression
        assert 255 in label or set(np.unique(label)).issubset({0, 1, 2, 3})

    def test_ndvi_bidirectional_suppresses_low_ndvi_irrigated(self, tmp_data_dir, rgb_band_config):
        """Irrigated pixels with NDVI < low_threshold are set to ignore (255)."""
        self._make_controlled_tile(tmp_data_dir)
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0],
            band_config=rgb_band_config,
            use_cache=False,
            noise_strategy="ndvi_bidirectional",
            ndvi_low_threshold=0.15,
            ndvi_high_threshold=0.4,
        )
        sample = ds[0]
        label = sample["label"].numpy()

        # Top-left: was irrigated (1) with NDVI=0.05 → should be 255
        assert (label[:112, :112] == 255).all()

        # Top-right: was irrigated (1) with NDVI=0.3 → should remain 1
        assert (label[:112, 112:] == 1).all()

    def test_ndvi_bidirectional_suppresses_high_ndvi_background(self, tmp_data_dir, rgb_band_config):
        """Background pixels with NDVI > high_threshold are set to ignore (255)."""
        self._make_controlled_tile(tmp_data_dir)
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0],
            band_config=rgb_band_config,
            use_cache=False,
            noise_strategy="ndvi_bidirectional",
            ndvi_low_threshold=0.15,
            ndvi_high_threshold=0.4,
        )
        sample = ds[0]
        label = sample["label"].numpy()

        # Bottom-right: was background (0) with NDVI=0.6 → should be 255
        assert (label[112:, 112:] == 255).all()

        # Bottom-left: was background (0) with NDVI=0.3 → should remain 0
        assert (label[112:, :112] == 0).all()

    def test_noise_strategy_none_leaves_labels_unchanged(self, tmp_data_dir, rgb_band_config):
        """Explicitly setting noise_strategy=None does nothing."""
        self._make_controlled_tile(tmp_data_dir)
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0],
            band_config=rgb_band_config,
            use_cache=False,
            noise_strategy=None,
        )
        sample = ds[0]
        label = sample["label"].numpy()

        # Top-left should still be 1 (not suppressed)
        assert (label[:112, :112] == 1).all()

    def test_ndvi_background_only_suppresses_high_ndvi_background(self, tmp_data_dir, rgb_band_config):
        """Background pixels with high NDVI are suppressed."""
        self._make_controlled_tile(tmp_data_dir)
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0],
            band_config=rgb_band_config,
            use_cache=False,
            noise_strategy="ndvi_background_only",
            ndvi_high_threshold=0.4,
        )
        sample = ds[0]
        label = sample["label"].numpy()

        # Bottom-right: was background (0) with NDVI=0.6 → should be 255
        assert (label[112:, 112:] == 255).all()

        # Bottom-left: was background (0) with NDVI=0.3 → should remain 0
        assert (label[112:, :112] == 0).all()

    def test_ndvi_background_only_leaves_irrigated_unchanged(self, tmp_data_dir, rgb_band_config):
        """Irrigated pixels with low NDVI are NOT suppressed (one-directional)."""
        self._make_controlled_tile(tmp_data_dir)
        ds = IrrigationDataset(
            data_root=tmp_data_dir,
            tile_ids=[0],
            band_config=rgb_band_config,
            use_cache=False,
            noise_strategy="ndvi_background_only",
            ndvi_high_threshold=0.4,
        )
        sample = ds[0]
        label = sample["label"].numpy()

        # Top-left: was irrigated (1) with NDVI=0.05 → should remain 1 (NOT suppressed)
        assert (label[:112, :112] == 1).all()

        # Top-right: was irrigated (1) with NDVI=0.3 → should remain 1
        assert (label[:112, 112:] == 1).all()


