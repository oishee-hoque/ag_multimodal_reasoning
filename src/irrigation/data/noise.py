"""
Label noise reduction utilities.

Two strategies implemented:
1. Polygon erosion — shrink label boundaries inward to remove edge noise
2. NDVI-based ignore masking — mark unlabeled high-NDVI pixels as ignore (255)
   instead of background (0) to avoid penalizing correct predictions

These are applied as label_transform functions passed to the dataset.
"""

import numpy as np
from scipy.ndimage import binary_erosion


def erode_labels(
    label: np.ndarray,
    image: np.ndarray,
    erosion_pixels: int = 2,
) -> np.ndarray:
    """
    Erode field label boundaries inward.

    For each irrigation class (1, 2, 3), create a binary mask,
    erode it, and set eroded-away pixels to ignore (255).
    Background (0) pixels are not eroded.

    Args:
        label: (224, 224) uint8 label array
        image: (C, 224, 224) float32 image array (unused here but
               kept for consistent label_transform signature)
        erosion_pixels: number of pixels to erode inward (2px = 20m at 10m res)

    Returns:
        Cleaned label array with eroded edges set to 255
    """
    cleaned = label.copy()
    struct = np.ones((2 * erosion_pixels + 1, 2 * erosion_pixels + 1))

    for cls_id in [1, 2, 3]:
        mask = label == cls_id
        if mask.sum() == 0:
            continue
        eroded = binary_erosion(mask, structure=struct)
        # Pixels that were in the class but got eroded away → ignore
        cleaned[(mask) & (~eroded)] = 255

    return cleaned


def ndvi_ignore_mask(
    label: np.ndarray,
    image: np.ndarray,
    ndvi_band_index: int = 9,
    ndvi_threshold: float = 0.4,
    seasons_axis_size: int = 1,
) -> np.ndarray:
    """
    Mark unlabeled high-NDVI pixels as ignore instead of background.

    If a pixel is labeled as background (0) but has high NDVI (suggesting
    it's actually an irrigated field without a label), set it to 255 so
    the model isn't penalized for correctly predicting irrigation there.

    For multi-season input, takes the max NDVI across seasons.

    Args:
        label: (224, 224) uint8 label array
        image: (C, 224, 224) float32 image array
        ndvi_band_index: index of NDVI median within each season's 14 bands
        ndvi_threshold: NDVI above this → likely irrigated
        seasons_axis_size: number of seasons in the stack (1 or 3)

    Returns:
        Label array with suspicious background pixels set to 255
    """
    cleaned = label.copy()

    # Extract NDVI band(s) from the stacked channels
    bands_per_season = image.shape[0] // seasons_axis_size
    ndvi_values = []
    for s in range(seasons_axis_size):
        start = s * bands_per_season
        ndvi_values.append(image[start + ndvi_band_index])

    # Max NDVI across available seasons
    max_ndvi = np.max(np.stack(ndvi_values), axis=0)  # (224, 224)

    # Mask: background pixels with high NDVI → ignore
    suspicious = (label == 0) & (max_ndvi > ndvi_threshold)
    cleaned[suspicious] = 255

    return cleaned


def combined_cleaning(
    label: np.ndarray,
    image: np.ndarray,
    erosion_pixels: int = 2,
    ndvi_band_index: int = 9,
    ndvi_threshold: float = 0.4,
    seasons_axis_size: int = 1,
) -> np.ndarray:
    """
    Apply both erosion and NDVI ignore masking.
    Erosion runs first, then NDVI masking.
    """
    label = erode_labels(label, image, erosion_pixels)
    label = ndvi_ignore_mask(
        label, image, ndvi_band_index, ndvi_threshold, seasons_axis_size
    )
    return label
