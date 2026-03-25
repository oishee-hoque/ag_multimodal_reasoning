"""Per-field feature extraction from multi-temporal Sentinel-2 imagery."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

BAND_NAMES = [
    "B02_Blue",
    "B03_Green",
    "B04_Red",
    "B08_NIR",
    "B8A_NarrowNIR",
    "B11_SWIR1_median",
    "B12_SWIR2_median",
    "B11_SWIR1_min",
    "B12_SWIR2_min",
    "NDVI_median",
    "NDVI_max",
    "NDMI_median",
    "NDMI_max",
    "valid_count",
]
SPECTRAL_BAND_INDICES = list(range(13))
KEY_TEMPORAL_BANDS = {
    "NDVI_med": 9,
    "NDVI_max": 10,
    "NDMI_med": 11,
    "NDMI_max": 12,
    "SWIR1_min": 7,
    "SWIR2_min": 8,
    "SWIR1_med": 5,
    "NIR": 3,
}
SEASONS = ["s3", "s4", "s5"]


def extract_shape_features(mask: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    n_pixels = int(mask.sum())
    features["area_pixels"] = float(n_pixels)
    features["area_hectares"] = float(n_pixels * 0.01)

    dilated = binary_dilation(mask)
    perimeter_pixels = int((dilated & ~mask).sum())
    features["perimeter_pixels"] = float(perimeter_pixels)
    features["compactness"] = float(4 * np.pi * n_pixels / (perimeter_pixels**2)) if perimeter_pixels > 0 else 0.0

    rows, cols = np.where(mask)
    if len(rows):
        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1
        features["bbox_height"] = float(height)
        features["bbox_width"] = float(width)
        features["bbox_aspect_ratio"] = float(height / max(width, 1))
        features["bbox_fill_ratio"] = float(n_pixels / max(height * width, 1))
    else:
        features.update({
            "bbox_height": 0.0,
            "bbox_width": 0.0,
            "bbox_aspect_ratio": 0.0,
            "bbox_fill_ratio": 0.0,
        })
    return features


def extract_spectral_features(images: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    for season_name in SEASONS:
        if season_name not in images:
            continue
        image = images[season_name]
        for band_idx in SPECTRAL_BAND_INDICES:
            pixels = image[band_idx][mask]
            if not len(pixels):
                continue
            prefix = f"{BAND_NAMES[band_idx]}_{season_name}"
            features[f"{prefix}_mean"] = float(pixels.mean())
            features[f"{prefix}_std"] = float(pixels.std())
            features[f"{prefix}_median"] = float(np.median(pixels))

    if "s4" in images:
        image_s4 = images["s4"]
        for name, band_idx in KEY_TEMPORAL_BANDS.items():
            pixels = image_s4[band_idx][mask]
            if len(pixels) > 1 and pixels.mean() != 0:
                features[f"{name}_spatial_cv"] = float(pixels.std() / abs(pixels.mean()))
            else:
                features[f"{name}_spatial_cv"] = 0.0
    return features


def extract_temporal_features(images: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    for name, band_idx in KEY_TEMPORAL_BANDS.items():
        season_means = []
        for season_name in SEASONS:
            if season_name in images:
                pixels = images[season_name][band_idx][mask]
                season_means.append(float(pixels.mean()) if len(pixels) else 0.0)

        if len(season_means) >= 2:
            features[f"{name}_temporal_mean"] = float(np.mean(season_means))
            features[f"{name}_temporal_std"] = float(np.std(season_means))
            features[f"{name}_temporal_range"] = float(max(season_means) - min(season_means))
            features[f"{name}_temporal_slope"] = float(season_means[-1] - season_means[0])
            features[f"{name}_temporal_peak_idx"] = float(np.argmax(season_means))

        season_stds = []
        for season_name in SEASONS:
            if season_name in images:
                pixels = images[season_name][band_idx][mask]
                if len(pixels) > 1:
                    season_stds.append(float(pixels.std()))
        if len(season_stds) >= 2:
            features[f"{name}_spatial_std_temporal_range"] = float(max(season_stds) - min(season_stds))
    return features


def extract_all_features(images: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float] | None:
    if int(mask.sum()) < 20:
        return None
    features: dict[str, float] = {}
    features.update(extract_shape_features(mask))
    features.update(extract_spectral_features(images, mask))
    features.update(extract_temporal_features(images, mask))
    return features
