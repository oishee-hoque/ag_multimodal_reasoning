"""
Label noise reduction utilities.

Strategies implemented:
1. Polygon erosion — shrink label boundaries inward to remove edge noise
2. NDVI-based ignore masking — mark unlabeled high-NDVI pixels as ignore (255)
3. Bidirectional NDVI masking — also marks labeled-but-fallow pixels as ignore
4. NDVI relabel — relabel low-NDVI irrigated pixels as background (0)

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


def ndvi_bidirectional_mask(
    label: np.ndarray,
    image: np.ndarray,
    ndvi_band_index: int = 9,
    high_threshold: float = 0.4,
    low_threshold: float = 0.15,
    seasons_axis_size: int = 1,
) -> np.ndarray:
    """
    Bidirectional NDVI-based label cleaning.

    Two types of noise are addressed:
    1. Unlabeled but irrigated: background pixels (class 0) with high NDVI
       are likely unlabeled irrigated fields → set to ignore (255)
    2. Labeled but not irrigated: irrigated pixels (class 1/2/3) with very
       low NDVI are likely mislabeled or fallow → set to ignore (255)

    Uses max NDVI across available seasons so that a field only needs to
    show high/low greenness in at least one season to be flagged.

    Args:
        label: (224, 224) uint8 label array
        image: (C, 224, 224) float32 image array
        ndvi_band_index: index of NDVI median within each season's 14 bands (default 9)
        high_threshold: NDVI above this for unlabeled pixels → likely irrigated → ignore
        low_threshold: NDVI below this for labeled irrigated pixels → likely mislabeled → ignore
        seasons_axis_size: number of seasons in the input stack (1 or 3)

    Returns:
        Cleaned label array with suspicious pixels set to 255 (ignore_index)
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

    # Direction 1: unlabeled but high NDVI → ignore
    suspicious_bg = (label == 0) & (max_ndvi > high_threshold)
    cleaned[suspicious_bg] = 255

    # Direction 2: labeled irrigated but low NDVI → ignore
    is_irrigated = (label == 1) | (label == 2) | (label == 3)
    suspicious_irr = is_irrigated & (max_ndvi < low_threshold)
    cleaned[suspicious_irr] = 255

    return cleaned


def ndvi_relabel_background(
    label: np.ndarray,
    image: np.ndarray,
    ndvi_band_index: int = 9,
    low_threshold: float = 0.15,
    seasons_axis_size: int = 1,
) -> np.ndarray:
    """
    Relabel low-NDVI irrigated pixels as background.

    Irrigated pixels (class 1/2/3) with max NDVI below low_threshold
    are reassigned to class 0 (background) instead of being ignored.

    Args:
        label: (224, 224) uint8 label array
        image: (C, 224, 224) float32 image array
        ndvi_band_index: index of NDVI median within each season's 14 bands (default 9)
        low_threshold: NDVI below this for labeled irrigated pixels → relabel as background
        seasons_axis_size: number of seasons in the input stack (1 or 3)

    Returns:
        Cleaned label array with low-NDVI irrigated pixels set to 0 (background)
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

    # Labeled irrigated but low NDVI → relabel as background
    is_irrigated = (label == 1) | (label == 2) | (label == 3)
    low_ndvi = is_irrigated & (max_ndvi < low_threshold)
    cleaned[low_ndvi] = 0

    return cleaned


# def combined_cleaning(
#     label: np.ndarray,
#     image: np.ndarray,
#     erosion_pixels: int = 2,
#     ndvi_band_index: int = 9,
#     high_threshold: float = 0.4,
#     low_threshold: float = 0.15,
#     seasons_axis_size: int = 1,
# ) -> np.ndarray:
#     """
#     Apply both erosion and bidirectional NDVI masking.
#     Erosion runs first, then NDVI masking.
#     """
#     label = erode_labels(label, image, erosion_pixels)
#     label = ndvi_bidirectional_mask(
#         label, image, ndvi_band_index, high_threshold, low_threshold, seasons_axis_size
#     )
#     return label
