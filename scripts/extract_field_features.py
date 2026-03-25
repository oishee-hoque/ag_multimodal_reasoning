"""Step 2: Extract per-field feature vectors from imagery + instance masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import rasterio
from tqdm import tqdm

from irrigation.field.feature_extraction import SEASONS, extract_all_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", type=str, required=True)
    parser.add_argument("--field_data_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--state_name", type=str, default="Unknown")
    args = parser.parse_args()

    state_path = Path(args.state_path)
    field_data_dir = Path(args.field_data_dir)
    mask_dir = field_data_dir / "field_masks"
    label_dir = field_data_dir / "field_labels"

    all_records = []
    for mask_path in tqdm(sorted(mask_dir.glob("tile_*_fields.tif")), desc="Extracting features"):
        tile_name = mask_path.stem.replace("_fields", "")
        tile_id = int(tile_name.split("_")[1])

        with rasterio.open(mask_path) as src:
            field_mask = src.read(1)

        label_path = label_dir / f"{tile_name}_labels.json"
        if not label_path.exists():
            continue
        with label_path.open() as f:
            field_labels = json.load(f)
        if not field_labels:
            continue

        images = {}
        for season in SEASONS:
            img_path = state_path / "images" / f"{tile_name}_{season}.tif"
            if img_path.exists():
                with rasterio.open(img_path) as src:
                    images[season] = src.read()
        if not images:
            continue

        for field_id_str, irr_class in field_labels.items():
            field_id = int(field_id_str)
            mask = field_mask == field_id
            features = extract_all_features(images, mask)
            if features is None:
                continue
            features["state"] = args.state_name
            features["tile_id"] = tile_id
            features["field_id"] = field_id
            features["label"] = int(irr_class)
            all_records.append(features)

    df = pd.DataFrame(all_records)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nExtracted features for {len(df)} fields")
    if len(df):
        print("Label distribution:")
        print(df["label"].value_counts().sort_index())
        print(f"\nFeature dimensions: {df.shape[1] - 4} features + 4 metadata columns")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
