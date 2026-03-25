"""Step 4: Run standalone SHAP analysis using saved model and feature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from irrigation.field.shap_analysis import run_shap_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--features_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    meta_cols = ["state", "tile_id", "field_id", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    model = xgb.XGBClassifier()
    model.load_model(args.model_path)

    labels = sorted(df["label"].unique().tolist())
    class_names = [f"Class_{i}" for i in labels]
    run_shap_analysis(model, X, feature_cols, class_names, Path(args.output_dir))


if __name__ == "__main__":
    main()
