"""Shared test fixtures for irrigation framework tests."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import tempfile
import shutil

from irrigation.data.bands import BandConfig, get_band_config


@pytest.fixture
def tmp_data_dir():
    """Create a temporary directory with synthetic GeoTIFF tiles."""
    tmpdir = Path(tempfile.mkdtemp())
    state_dir = tmpdir / "TestState"
    images_dir = state_dir / "images"
    labels_dir = state_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    num_tiles = 10
    h, w = 224, 224
    num_bands = 14
    seasons = ["s3", "s4", "s5"]

    transform = from_bounds(0, 0, 1, 1, w, h)

    for tid in range(num_tiles):
        tile_name = f"tile_{tid:04d}"

        # Create label: random classes 0-3, with some 255 (ignore)
        rng = np.random.RandomState(tid)
        label = rng.choice([0, 1, 2, 3], size=(h, w)).astype(np.uint8)
        # Set some pixels to ignore
        label[0:10, 0:10] = 255

        label_path = labels_dir / f"{tile_name}_label.tif"
        with rasterio.open(
            label_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="uint8",
            transform=transform,
            crs="EPSG:4326",
        ) as dst:
            dst.write(label, 1)

        # Create image for each season
        for season in seasons:
            img_data = rng.rand(num_bands, h, w).astype(np.float32)
            # Make NDVI band (index 9) realistic
            img_data[9] = rng.uniform(0.1, 0.8, size=(h, w)).astype(np.float32)

            img_path = images_dir / f"{tile_name}_{season}.tif"
            with rasterio.open(
                img_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=num_bands,
                dtype="float32",
                transform=transform,
                crs="EPSG:4326",
            ) as dst:
                dst.write(img_data)

    # Create metadata.csv for spatial block split tests
    import pandas as pd

    rows = []
    for tid in range(num_tiles):
        for season in seasons:
            rows.append({
                "tile_id": tid,
                "season": season,
                "center_lon": -105.0 + tid * 0.5,
                "center_lat": 39.0 + tid * 0.3,
            })
    pd.DataFrame(rows).to_csv(state_dir / "metadata.csv", index=False)

    yield state_dir

    shutil.rmtree(tmpdir)


@pytest.fixture
def rgb_band_config():
    """RGB band config for testing."""
    return get_band_config("rgb_s4")


@pytest.fixture
def spectral_band_config():
    """Spectral band config for testing."""
    return get_band_config("spectral_s4")


@pytest.fixture
def temporal_band_config():
    """Temporal band config for testing."""
    return get_band_config("temporal")
