"""Lightning DataModule for field-level feature classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from irrigation.field.field_dataset import FieldFeatureDataset


class FieldDataModule(pl.LightningDataModule):
    """Build train/val/test loaders from field-feature CSV files."""

    def __init__(
        self,
        train_csv: str | Path,
        test_csv: str | Path | None = None,
        val_fraction: float = 0.15,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv) if test_csv else None
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        self.seed = seed

    def setup(self, stage: str | None = None):
        train_df = pd.read_csv(self.train_csv)
        if self.test_csv is None:
            train_df, test_df = train_test_split(
                train_df,
                test_size=0.2,
                stratify=train_df["label"],
                random_state=self.seed,
            )
        else:
            test_df = pd.read_csv(self.test_csv)

        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_fraction,
            stratify=train_df["label"],
            random_state=self.seed,
        )

        tmp_dir = self.train_csv.parent / ".field_splits"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        train_path = tmp_dir / "train.csv"
        val_path = tmp_dir / "val.csv"
        test_path = tmp_dir / "test.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        self.train_dataset = FieldFeatureDataset(train_path, normalize=self.normalize)
        stats = self.train_dataset.get_stats() if self.normalize else None

        self.val_dataset = FieldFeatureDataset(
            val_path,
            feature_columns=self.train_dataset.feature_columns,
            normalize=self.normalize,
            stats=stats,
        )
        self.test_dataset = FieldFeatureDataset(
            test_path,
            feature_columns=self.train_dataset.feature_columns,
            normalize=self.normalize,
            stats=stats,
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
            persistent_workers=self.num_workers > 0,
        )

    @property
    def num_features(self) -> int:
        return self.train_dataset.num_features

    @property
    def num_classes(self) -> int:
        return self.train_dataset.num_classes

    def get_class_weights(self):
        return self.train_dataset.get_class_weights()
