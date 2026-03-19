"""
Lightning DataModule for irrigation dataset.

Handles:
- Loading tile IDs from metadata CSVs
- Creating train/val/test splits (cross-state or within-state)
- Instantiating datasets with correct band configs and transforms
- DataLoader creation with proper settings for HPC
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

from irrigation.data.dataset import IrrigationDataset
from irrigation.data.bands import get_band_config
from irrigation.data.transforms import get_train_transforms, get_val_transforms
from irrigation.data.normalization import compute_band_statistics
from irrigation.data.noise import (
    ndvi_bidirectional_mask,
    ndvi_ignore_mask,
    ndvi_relabel_background,
)


NOISE_FUNCTIONS = {
    "none": None,
    "ndvi_mask": ndvi_ignore_mask,
    "ndvi_bidirectional": ndvi_bidirectional_mask,
    "ndvi_relabel": ndvi_relabel_background,
}


class IrrigationDataModule(pl.LightningDataModule):
    """
    DataModule supporting two split modes:

    1. cross_state: Train on state A, test on state B
       - train_state: path to training state data
       - test_state: path to test state data
       - val_fraction: fraction of train_state held out for validation

    2. within_state: Random spatial split within one state
       - train_state: path to the state data
       - test_state: not used
       - val_fraction + test_fraction define splits
    """

    def __init__(
        self,
        train_state_path: str,
        test_state_path: str | None,
        band_group: str,
        split_mode: str = "cross_state",  # "cross_state" or "within_state"
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,  # only for within_state
        noise_strategy: str = "none",  # "none", "ndvi_mask", "ndvi_bidirectional", "ndvi_relabel"
        high_threshold: float = 0.4,
        low_threshold: float = 0.15,
        ndvi_threshold: float = 0.4,  # backward compat for ndvi_mask
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_state_path = Path(train_state_path)
        self.test_state_path = Path(test_state_path) if test_state_path else None
        self.band_config = get_band_config(band_group)
        self.split_mode = split_mode
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        # Configure noise cleaning
        self.noise_strategy = noise_strategy
        self.label_transform_fn = NOISE_FUNCTIONS.get(noise_strategy)

        # If the noise function needs extra args, wrap it
        if noise_strategy == "ndvi_mask":
            _ndvi_thresh = ndvi_threshold
            _n_seasons = len(self.band_config.seasons)
            self.label_transform_fn = lambda l, i: ndvi_ignore_mask(
                l, i, ndvi_threshold=_ndvi_thresh, seasons_axis_size=_n_seasons
            )
        elif noise_strategy == "ndvi_bidirectional":
            _high_thresh = high_threshold
            _low_thresh = low_threshold
            _n_seasons = len(self.band_config.seasons)
            self.label_transform_fn = lambda l, i: ndvi_bidirectional_mask(
                l, i,
                ndvi_band_index=9,
                high_threshold=_high_thresh,
                low_threshold=_low_thresh,
                seasons_axis_size=_n_seasons,
            )
        elif noise_strategy == "ndvi_relabel":
            _low_thresh = low_threshold
            _n_seasons = len(self.band_config.seasons)
            self.label_transform_fn = lambda l, i: ndvi_relabel_background(
                l, i,
                ndvi_band_index=9,
                low_threshold=_low_thresh,
                seasons_axis_size=_n_seasons,
            )

    def _get_tile_ids(self, state_path: Path) -> list[int]:
        """Extract tile IDs from the labels directory."""
        label_dir = state_path / "labels"
        tile_ids = []
        for f in sorted(label_dir.glob("tile_*_label.tif")):
            # Parse tile_0042_label.tif → 42
            tid = int(f.stem.split("_")[1])
            tile_ids.append(tid)
        return tile_ids

    def _spatial_block_split(
        self,
        tile_ids: list[int],
        state_path: Path,
        val_frac: float,
        test_frac: float,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Split tiles into train/val/test using spatial blocking.

        Uses tile center coordinates from metadata.csv to create
        spatial clusters, then assigns clusters (not individual tiles)
        to splits. This prevents spatial autocorrelation leakage.
        """
        metadata = pd.read_csv(state_path / "metadata.csv")

        # Get unique tiles with their coordinates
        # metadata may have multiple rows per tile (one per season)
        tile_meta = metadata.drop_duplicates(subset="tile_id")[
            ["tile_id", "center_lon", "center_lat"]
        ]
        tile_meta = tile_meta[tile_meta["tile_id"].isin(tile_ids)]

        # Create spatial blocks by binning coordinates into grid cells
        n_blocks = max(10, len(tile_ids) // 20)  # ~20 tiles per block
        lon_bins = pd.qcut(
            tile_meta["center_lon"],
            q=int(np.sqrt(n_blocks)),
            labels=False,
            duplicates="drop",
        )
        lat_bins = pd.qcut(
            tile_meta["center_lat"],
            q=int(np.sqrt(n_blocks)),
            labels=False,
            duplicates="drop",
        )
        tile_meta = tile_meta.copy()
        tile_meta["block"] = lon_bins.astype(str) + "_" + lat_bins.astype(str)

        # Split blocks into train/val/test
        blocks = tile_meta["block"].values
        tile_id_arr = tile_meta["tile_id"].values

        # First split: separate test
        splitter1 = GroupShuffleSplit(
            n_splits=1, test_size=test_frac, random_state=self.seed
        )
        trainval_idx, test_idx = next(splitter1.split(tile_id_arr, groups=blocks))

        # Second split: separate val from train
        trainval_blocks = blocks[trainval_idx]
        trainval_tiles = tile_id_arr[trainval_idx]
        relative_val_frac = val_frac / (1 - test_frac)
        splitter2 = GroupShuffleSplit(
            n_splits=1, test_size=relative_val_frac, random_state=self.seed
        )
        train_idx, val_idx = next(
            splitter2.split(trainval_tiles, groups=trainval_blocks)
        )

        return (
            trainval_tiles[train_idx].tolist(),
            trainval_tiles[val_idx].tolist(),
            tile_id_arr[test_idx].tolist(),
        )

    def _compute_normalization_stats(
        self, train_ids: list[int], data_root: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-band normalization statistics from training tiles only.

        Always uses dataset-specific statistics (computed from training set),
        even for RGB with pretrained encoders. This is best practice for
        satellite imagery because:
        - Sentinel-2 reflectance distributions differ fundamentally from
          ImageNet natural photos (different value ranges, spectral response)
        - Ensures zero-mean, unit-variance inputs regardless of data scale
        - Prevents data leakage (only training tiles contribute)
        """
        print(f"Computing per-band normalization statistics from {len(train_ids)} training tiles...")
        mean, std = compute_band_statistics(data_root, train_ids, self.band_config)
        print(f"  Band means: {mean}")
        print(f"  Band stds:  {std}")
        return mean, std

    def setup(self, stage: str | None = None):
        """Create train/val/test datasets based on split mode."""

        if self.split_mode == "cross_state":
            # Train + val from train_state, test from test_state
            all_train_ids = self._get_tile_ids(self.train_state_path)

            # Simple random split for val (spatial blocking less critical
            # when test is entirely different state)
            rng = np.random.RandomState(self.seed)
            rng.shuffle(all_train_ids)
            n_val = int(len(all_train_ids) * self.val_fraction)
            val_ids = all_train_ids[:n_val]
            train_ids = all_train_ids[n_val:]

            test_ids = self._get_tile_ids(self.test_state_path)

            # Compute normalization stats from training tiles only
            norm_mean, norm_std = self._compute_normalization_stats(
                train_ids, self.train_state_path
            )

            self.train_dataset = IrrigationDataset(
                self.train_state_path,
                train_ids,
                self.band_config,
                transform=get_train_transforms(),
                label_transform=self.label_transform_fn,
                normalize_mean=norm_mean,
                normalize_std=norm_std,
            )
            self.val_dataset = IrrigationDataset(
                self.train_state_path,
                val_ids,
                self.band_config,
                transform=get_val_transforms(),
                label_transform=None,
                normalize_mean=norm_mean,
                normalize_std=norm_std,
            )
            self.test_dataset = IrrigationDataset(
                self.test_state_path,
                test_ids,
                self.band_config,
                transform=get_val_transforms(),
                label_transform=None,
                normalize_mean=norm_mean,
                normalize_std=norm_std,
            )

        elif self.split_mode == "within_state":
            all_ids = self._get_tile_ids(self.train_state_path)
            train_ids, val_ids, test_ids = self._spatial_block_split(
                all_ids,
                self.train_state_path,
                self.val_fraction,
                self.test_fraction,
            )

            # Compute normalization stats from training tiles only
            norm_mean, norm_std = self._compute_normalization_stats(
                train_ids, self.train_state_path
            )

            self.train_dataset = IrrigationDataset(
                self.train_state_path,
                train_ids,
                self.band_config,
                transform=get_train_transforms(),
                label_transform=self.label_transform_fn,
                normalize_mean=norm_mean,
                normalize_std=norm_std,
            )
            self.val_dataset = IrrigationDataset(
                self.train_state_path,
                val_ids,
                self.band_config,
                transform=get_val_transforms(),
                label_transform=None,
                normalize_mean=norm_mean,
                normalize_std=norm_std,
            )
            self.test_dataset = IrrigationDataset(
                self.train_state_path,
                test_ids,
                self.band_config,
                transform=get_val_transforms(),
                label_transform=None,
                normalize_mean=norm_mean,
                normalize_std=norm_std,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
