"""
Compute class weights from label data for weighted loss.

Supports two strategies:
- inverse_frequency: weight = median_freq / class_freq (effective for imbalanced segmentation)
- inverse_sqrt: weight = 1 / sqrt(class_freq) (gentler rebalancing)

Both strategies normalize weights so the minimum non-zero weight is 1.0.
"""

import numpy as np
import rasterio
from pathlib import Path


CLASS_NAMES = ["background", "flood", "sprinkler", "drip"]


def compute_class_frequencies(
    state_path: Path,
    tile_ids: list[int] | None = None,
    num_classes: int = 4,
    ignore_index: int = 255,
) -> np.ndarray:
    """
    Count pixels per class across all label tiles.

    Args:
        state_path: Path to state directory containing labels/
        tile_ids: Optional subset of tiles. If None, uses all.
        num_classes: Number of valid classes
        ignore_index: Label value to exclude from counts

    Returns:
        (num_classes,) array of pixel counts per class
    """
    label_dir = state_path / "labels"

    if tile_ids is None:
        label_files = sorted(label_dir.glob("tile_*_label.tif"))
    else:
        label_files = [label_dir / f"tile_{tid:04d}_label.tif" for tid in tile_ids]

    counts = np.zeros(num_classes, dtype=np.int64)

    for fpath in label_files:
        if not fpath.exists():
            continue
        with rasterio.open(fpath) as src:
            label = src.read(1)
        for cls in range(num_classes):
            counts[cls] += (label == cls).sum()

    return counts


def compute_inverse_frequency_weights(
    counts: np.ndarray,
    zero_classes: list[int] | None = None,
) -> np.ndarray:
    """
    Compute inverse-frequency weights (median frequency balancing).

    weight_c = median(freq) / freq_c

    Args:
        counts: (num_classes,) pixel counts
        zero_classes: Class indices to set weight=0 (ignored in loss)

    Returns:
        (num_classes,) float32 weight array
    """
    total = counts.sum()
    if total == 0:
        return np.ones(len(counts), dtype=np.float32)

    freq = counts / total
    # Avoid division by zero for classes with no pixels
    freq = np.where(freq > 0, freq, 1.0)
    median_freq = np.median(freq[counts > 0])
    weights = (median_freq / freq).astype(np.float32)

    if zero_classes:
        for cls in zero_classes:
            weights[cls] = 0.0

    # Normalize so min non-zero weight is 1.0
    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        weights = weights / nonzero.min()

    return weights


def compute_inverse_sqrt_weights(
    counts: np.ndarray,
    zero_classes: list[int] | None = None,
) -> np.ndarray:
    """
    Compute inverse-sqrt-frequency weights (gentler rebalancing).

    weight_c = 1 / sqrt(freq_c)

    Args:
        counts: (num_classes,) pixel counts
        zero_classes: Class indices to set weight=0 (ignored in loss)

    Returns:
        (num_classes,) float32 weight array
    """
    total = counts.sum()
    if total == 0:
        return np.ones(len(counts), dtype=np.float32)

    freq = counts / total
    freq = np.where(freq > 0, freq, 1.0)
    weights = (1.0 / np.sqrt(freq)).astype(np.float32)

    if zero_classes:
        for cls in zero_classes:
            weights[cls] = 0.0

    # Normalize so min non-zero weight is 1.0
    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        weights = weights / nonzero.min()

    return weights


def print_class_weight_summary(
    counts: np.ndarray,
    weights: np.ndarray,
    strategy: str,
) -> None:
    """Print a formatted summary of class distribution and weights."""
    total = counts.sum()
    print(f"\n{'='*60}")
    print(f"Class Weight Summary (strategy: {strategy})")
    print(f"{'='*60}")
    print(f"{'Class':<15} {'Pixels':>12} {'Fraction':>10} {'Weight':>10}")
    print(f"{'-'*47}")
    for i, name in enumerate(CLASS_NAMES):
        frac = counts[i] / total if total > 0 else 0
        w = weights[i]
        status = " (IGNORED)" if w == 0 else ""
        print(f"{name:<15} {counts[i]:>12,} {frac:>10.4f} {w:>10.4f}{status}")
    print(f"{'='*60}\n")
