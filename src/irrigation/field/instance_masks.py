"""Generate per-tile field instance masks from government geodatabase polygons."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

IRRIGATION_CLASS_MAP = {
    "flood": 1,
    "furrow": 1,
    "gated pipe": 1,
    "gated_pipe": 1,
    "subirrigated": 1,
    "sprinkler": 2,
    "center pivot": 2,
    "center_pivot": 2,
    "drip": 3,
    "micro": 3,
}


def _normalize_text(value: str) -> str:
    return value.lower().strip().replace("-", " ").replace("_", " ")


def _select_polygon_layer(gdb_path: Path) -> str:
    import fiona

    layers = fiona.listlayers(gdb_path)
    selected_layer = None
    for layer in layers:
        sample = gpd.read_file(gdb_path, layer=layer, rows=1)
        if len(sample) == 0:
            continue
        geom_type = sample.geometry.iloc[0].geom_type
        if geom_type in ("Polygon", "MultiPolygon"):
            if "2023" in layer.lower() or "2022" in layer.lower() or "2024" in layer.lower():
                return layer
            if selected_layer is None:
                selected_layer = layer
    if selected_layer is None:
        raise ValueError(f"No polygon layer found in {gdb_path}")
    return selected_layer


def _detect_irrigation_column(gdf: gpd.GeoDataFrame) -> str:
    known_types = set(IRRIGATION_CLASS_MAP.keys()) | {k.replace("_", " ") for k in IRRIGATION_CLASS_MAP}
    best_col = None
    best_count = 0

    for col in gdf.select_dtypes(include="object").columns:
        normalized = gdf[col].astype(str).map(_normalize_text)
        matches = normalized.isin(known_types).sum()
        if matches > best_count:
            best_count = int(matches)
            best_col = col

    if best_col is None or best_count == 0:
        raise ValueError("Could not auto-detect irrigation-type column in geodatabase layer")

    print(f"Detected irrigation type column: {best_col} ({best_count} direct matches)")
    return best_col


def load_irrigation_polygons(gdb_path: str | Path) -> gpd.GeoDataFrame:
    """Load irrigation polygons from a .gdb file with standardized class IDs."""
    gdb_path = Path(gdb_path)
    selected_layer = _select_polygon_layer(gdb_path)
    print(f"Selected geodatabase layer: {selected_layer}")

    gdf = gpd.read_file(gdb_path, layer=selected_layer)
    irr_col = _detect_irrigation_column(gdf)

    normalized_types = gdf[irr_col].astype(str).map(_normalize_text)
    map_dict = {k.replace("_", " "): v for k, v in IRRIGATION_CLASS_MAP.items()}
    map_dict.update({k: v for k, v in IRRIGATION_CLASS_MAP.items()})

    gdf = gdf.copy()
    gdf["irrigation_class"] = normalized_types.map(map_dict).fillna(0).astype(int)
    gdf = gdf[gdf["irrigation_class"] > 0].copy()
    return gdf


def create_instance_mask_for_tile(
    polygons: gpd.GeoDataFrame,
    tile_bounds: tuple[float, float, float, float],
    tile_crs: str,
    tile_shape: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, dict[int, int]]:
    """Create an instance mask and field-label mapping for one tile."""
    xmin, ymin, xmax, ymax = tile_bounds
    tile_box = box(xmin, ymin, xmax, ymax)

    if polygons.crs is None:
        raise ValueError("Input polygons must have a CRS")
    if polygons.crs.to_string() != tile_crs:
        polygons = polygons.to_crs(tile_crs)

    clipped = polygons.clip(tile_box)
    clipped = clipped[~clipped.geometry.is_empty]

    if len(clipped) == 0:
        return np.zeros(tile_shape, dtype=np.int32), {}

    transform = from_bounds(xmin, ymin, xmax, ymax, tile_shape[1], tile_shape[0])
    field_mask = np.zeros(tile_shape, dtype=np.int32)
    field_labels: dict[int, int] = {}

    for idx, (_, row) in enumerate(clipped.iterrows()):
        field_id = idx + 1
        single_mask = rasterize(
            [(row.geometry, field_id)],
            out_shape=tile_shape,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=np.int32,
        )

        new_pixels = (single_mask > 0) & (field_mask == 0)
        field_mask[new_pixels] = field_id
        if int(new_pixels.sum()) > 0:
            field_labels[field_id] = int(row["irrigation_class"])

    return field_mask, field_labels


def generate_all_instance_masks(
    state_path: Path,
    gdb_path: Path,
    output_dir: Path,
    min_field_pixels: int = 20,
) -> dict[str, int]:
    """Generate instance masks/labels for all tiles in a state directory."""
    import pandas as pd
    from tqdm import tqdm

    polygons = load_irrigation_polygons(gdb_path)
    metadata = pd.read_csv(state_path / "metadata.csv")
    tile_meta = metadata.drop_duplicates(subset="tile_id")

    mask_dir = output_dir / "field_masks"
    label_dir = output_dir / "field_labels"
    mask_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total_tiles": 0, "tiles_with_fields": 0, "total_fields": 0}

    for _, row in tqdm(tile_meta.iterrows(), total=len(tile_meta), desc="Generating masks"):
        tile_id = int(row["tile_id"])
        tile_name = f"tile_{tile_id:04d}"
        tile_bounds = (row["utm_xmin"], row["utm_ymin"], row["utm_xmax"], row["utm_ymax"])
        tile_crs = f"EPSG:{int(row['utm_epsg'])}"

        field_mask, field_labels = create_instance_mask_for_tile(polygons, tile_bounds, tile_crs)

        for fid in list(field_labels.keys()):
            if int((field_mask == fid).sum()) < min_field_pixels:
                field_mask[field_mask == fid] = 0
                del field_labels[fid]

        stats["total_tiles"] += 1
        if field_labels:
            stats["tiles_with_fields"] += 1
            stats["total_fields"] += len(field_labels)

        img_path = state_path / "images" / f"{tile_name}_s4.tif"
        if img_path.exists():
            with rasterio.open(img_path) as src:
                profile = src.profile.copy()
            profile.update(dtype="int32", count=1, compress="deflate")
            with rasterio.open(mask_dir / f"{tile_name}_fields.tif", "w", **profile) as dst:
                dst.write(field_mask[np.newaxis, :, :])

        with (label_dir / f"{tile_name}_labels.json").open("w") as f:
            json.dump({str(k): int(v) for k, v in field_labels.items()}, f)

    print(f"Generated masks for {stats['total_tiles']} tiles")
    print(f"  Tiles with fields: {stats['tiles_with_fields']}")
    print(f"  Total fields: {stats['total_fields']}")
    return stats
