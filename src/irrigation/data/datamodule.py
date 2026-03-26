"""
Lightning DataModule for irrigation dataset.

Handles:
- Loading tile IDs from metadata CSVs
- Creating train/val/test splits (cross-state or within-state)
- Instantiating datasets with correct band configs and transforms
- DataLoader creation with proper settings for HPC

Note: Raw pixel values are used directly (no per-band normalization).
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
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        noise_strategy: str | None = None,
        ndvi_high_threshold: float = 0.4,
        ndvi_low_threshold: float = 0.15,
        ndvi_band_index: int = 9,
        ndvi_season: str = "s4",
        use_field_channels: bool = False,
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
        self.noise_strategy = noise_strategy
        self.ndvi_high_threshold = ndvi_high_threshold
        self.ndvi_low_threshold = ndvi_low_threshold
        self.ndvi_band_index = ndvi_band_index
        self.ndvi_season = ndvi_season
        self.use_field_channels = use_field_channels

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

            # Noise refinement only for training data
            noise_kwargs = dict(
                noise_strategy=self.noise_strategy,
                ndvi_high_threshold=self.ndvi_high_threshold,
                ndvi_low_threshold=self.ndvi_low_threshold,
                ndvi_band_index=self.ndvi_band_index,
                ndvi_season=self.ndvi_season,
            )

            self.train_dataset = IrrigationDataset(
                self.train_state_path,
                train_ids,
                self.band_config,
                transform=get_train_transforms(),
                use_field_channels=self.use_field_channels,
                **noise_kwargs,
            )
            self.val_dataset = IrrigationDataset(
                self.train_state_path,
                val_ids,
                self.band_config,
                transform=get_val_transforms(),
                use_field_channels=self.use_field_channels,
            )
            self.test_dataset = IrrigationDataset(
                self.test_state_path,
                test_ids,
                self.band_config,
                transform=get_val_transforms(),
                use_field_channels=self.use_field_channels,
            )

        elif self.split_mode == "within_state":
            all_ids = self._get_tile_ids(self.train_state_path)
            train_ids, val_ids, test_ids = self._spatial_block_split(
                all_ids,
                self.train_state_path,
                self.val_fraction,
                self.test_fraction,
            )

            noise_kwargs = dict(
                noise_strategy=self.noise_strategy,
                ndvi_high_threshold=self.ndvi_high_threshold,
                ndvi_low_threshold=self.ndvi_low_threshold,
                ndvi_band_index=self.ndvi_band_index,
                ndvi_season=self.ndvi_season,
            )

            self.train_dataset = IrrigationDataset(
                self.train_state_path,
                train_ids,
                self.band_config,
                transform=get_train_transforms(),
                use_field_channels=self.use_field_channels,
                **noise_kwargs,
            )
            self.val_dataset = IrrigationDataset(
                self.train_state_path,
                val_ids,
                self.band_config,
                transform=get_val_transforms(),
                use_field_channels=self.use_field_channels,
            )
            self.test_dataset = IrrigationDataset(
                self.train_state_path,
                test_ids,
                self.band_config,
                transform=get_val_transforms(),
                use_field_channels=self.use_field_channels,
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
