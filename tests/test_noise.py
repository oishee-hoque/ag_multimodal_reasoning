"""Tests for bidirectional NDVI label cleaning."""

import numpy as np
from irrigation.data.noise import ndvi_bidirectional_mask


def test_bidirectional_mask():
    """Test that bidirectional NDVI masking works correctly."""
    # Create a simple 4x4 label and image
    label = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 0, 0, 3],
        [0, 0, 0, 0],
    ], dtype=np.uint8)

    # Create image with NDVI at band index 9
    # 14 bands, 4x4 spatial
    image = np.zeros((14, 4, 4), dtype=np.float32)

    # Set NDVI values:
    # - (0,0) background with high NDVI=0.6 → should become ignore
    # - (0,2) irrigated with low NDVI=0.05 → should become ignore
    # - (0,3) irrigated with normal NDVI=0.5 → should stay
    # - (2,3) irrigated with low NDVI=0.1 → should become ignore
    # - (3,0) background with low NDVI=0.1 → should stay background
    image[9, 0, 0] = 0.6   # high NDVI, unlabeled
    image[9, 0, 2] = 0.05  # low NDVI, labeled flood
    image[9, 0, 3] = 0.5   # normal NDVI, labeled flood
    image[9, 1, 2] = 0.5   # normal NDVI, labeled flood
    image[9, 1, 3] = 0.5   # normal NDVI, labeled sprinkler
    image[9, 2, 3] = 0.1   # low NDVI, labeled drip
    image[9, 3, 0] = 0.1   # low NDVI, background

    result = ndvi_bidirectional_mask(
        label, image,
        ndvi_band_index=9,
        high_threshold=0.4,
        low_threshold=0.15,
        seasons_axis_size=1,
    )

    # Background + high NDVI → ignore
    assert result[0, 0] == 255
    # Irrigated + very low NDVI → ignore
    assert result[0, 2] == 255
    assert result[2, 3] == 255
    # Irrigated + normal NDVI → unchanged
    assert result[0, 3] == 1
    assert result[1, 2] == 1
    assert result[1, 3] == 2
    # Background + low NDVI → unchanged (stays 0)
    assert result[3, 0] == 0


def test_bidirectional_mask_multitemporal():
    """Test with 3 seasons — should use max NDVI across seasons."""
    label = np.array([[1, 0]], dtype=np.uint8)

    # 42 channels (14 bands × 3 seasons), 1×2 spatial
    image = np.zeros((42, 1, 2), dtype=np.float32)

    # Field at (0,0) labeled irrigated:
    # season 1 NDVI = 0.05 (low), season 2 = 0.5 (high), season 3 = 0.3
    # Max = 0.5, so should NOT be masked (field was active in season 2)
    image[9, 0, 0] = 0.05    # season 1, band 9
    image[23, 0, 0] = 0.5    # season 2, band 9 (offset by 14)
    image[37, 0, 0] = 0.3    # season 3, band 9 (offset by 28)

    result = ndvi_bidirectional_mask(
        label, image,
        ndvi_band_index=9,
        high_threshold=0.4,
        low_threshold=0.15,
        seasons_axis_size=3,
    )

    # Should stay labeled because max NDVI across seasons = 0.5
    assert result[0, 0] == 1
