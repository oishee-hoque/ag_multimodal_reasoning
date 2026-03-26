"""
Field-aware auxiliary channels derived from label masks.

These channels encode spatial field structure that helps the model
understand where fields are and how they differ from each other.

Three channels are produced:
1. Boundary map: binary edges of field polygons
2. Distance transform: normalized distance from each pixel to nearest field edge
3. Field size encoding: each pixel gets the normalized log-area of its field

These are concatenated to the spectral bands as additional input channels.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label as ndimage_label


def create_boundary_channel(label: np.ndarray) -> np.ndarray:
    """
    Create a binary boundary map from the label mask.

    Boundaries are the 1-pixel-wide edges of each labeled field polygon.
    Background boundaries are NOT included — only irrigation field edges.

    Args:
        label: (H, W) uint8 label array with classes 0-3 (and possibly 255)

    Returns:
        (H, W) float32 array, 1.0 at field boundaries, 0.0 elsewhere
    """
    boundary = np.zeros(label.shape, dtype=np.float32)

    for cls in [1, 2, 3]:
        mask = (label == cls)
        if mask.sum() == 0:
            continue
        dilated = binary_dilation(mask, iterations=1)
        eroded = binary_erosion(mask, iterations=1)
        # Handle very small fields where erosion removes everything
        if eroded.sum() == 0:
            eroded = mask
        edges = dilated ^ eroded  # XOR gives the boundary ring
        boundary[edges] = 1.0

    return boundary


def create_distance_channel(label: np.ndarray) -> np.ndarray:
    """
    Create a normalized distance-to-boundary channel.

    For each labeled pixel, computes the Euclidean distance to the nearest
    edge of its field. Pixels at field centers have high values, pixels near
    edges have low values. Background pixels are 0.

    This gives the model a smooth signal about field interior vs edge.

    Args:
        label: (H, W) uint8 label array

    Returns:
        (H, W) float32 array, normalized to [0, 1]
    """
    irrigated_mask = (label == 1) | (label == 2) | (label == 3)

    if irrigated_mask.sum() == 0:
        return np.zeros(label.shape, dtype=np.float32)

    # Distance transform on the combined irrigated mask
    dist = distance_transform_edt(irrigated_mask).astype(np.float32)

    # Normalize to [0, 1]
    max_dist = dist.max()
    if max_dist > 0:
        dist = dist / max_dist

    return dist


def create_field_size_channel(label: np.ndarray) -> np.ndarray:
    """
    Create a field size encoding channel.

    Each pixel gets the normalized log-area of the connected component
    (field) it belongs to. This tells the model whether a pixel is part
    of a large or small field, which correlates with irrigation type
    (center pivots tend to be larger than flood-irrigated parcels).

    Args:
        label: (H, W) uint8 label array

    Returns:
        (H, W) float32 array, normalized to [0, 1]
    """
    size_channel = np.zeros(label.shape, dtype=np.float32)

    irrigated_mask = (label == 1) | (label == 2) | (label == 3)

    if irrigated_mask.sum() == 0:
        return size_channel

    # Label connected components (individual fields)
    labeled_fields, num_fields = ndimage_label(irrigated_mask)

    for field_id in range(1, num_fields + 1):
        field_pixels = (labeled_fields == field_id)
        area = field_pixels.sum()
        size_channel[field_pixels] = np.log1p(area)

    # Normalize to [0, 1]
    max_val = size_channel.max()
    if max_val > 0:
        size_channel = size_channel / max_val

    return size_channel


def create_all_field_channels(label: np.ndarray) -> np.ndarray:
    """
    Create all three field-aware channels.

    Args:
        label: (H, W) uint8 label array

    Returns:
        (3, H, W) float32 array:
            channel 0: boundary map
            channel 1: distance transform
            channel 2: field size encoding
    """
    boundary = create_boundary_channel(label)
    distance = create_distance_channel(label)
    size_enc = create_field_size_channel(label)

    return np.stack([boundary, distance, size_enc], axis=0)
