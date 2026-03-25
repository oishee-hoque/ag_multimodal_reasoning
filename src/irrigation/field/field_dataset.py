"""PyTorch Dataset for field-level classification from feature CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FieldFeatureDataset(Dataset):
    """Dataset for field-level classification from pre-extracted features."""

    def __init__(
        self,
        csv_path: str | Path,
        feature_columns: list[str] | None = None,
        label_column: str = "label",
        meta_columns: list[str] | None = None,
        normalize: bool = True,
        stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        self.meta_columns = meta_columns or ["state", "tile_id", "field_id"]

        exclude = set([label_column] + self.meta_columns)
        self.feature_columns = [c for c in self.df.columns if c not in exclude] if feature_columns is None else feature_columns

        self.features = self.df[self.feature_columns].values.astype(np.float32)
        self.labels = self.df[label_column].values.astype(np.int64)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

        self.normalize = normalize
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                self.mean = self.features.mean(axis=0)
                self.std = self.features.std(axis=0)
                self.std[self.std < 1e-8] = 1.0
            self.features = (self.features - self.mean) / self.std

    def get_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.std is None:
            raise ValueError("Normalization stats are unavailable because normalize=False")
        return self.mean, self.std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "label": torch.tensor(self.labels[idx]),
        }

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)

    @property
    def num_classes(self) -> int:
        return len(np.unique(self.labels))

    def get_class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(counts)
        return torch.from_numpy(weights).float()
