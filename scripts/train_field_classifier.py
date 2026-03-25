"""Step 3: Train field-level classifiers and evaluate."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from irrigation.field.classifiers import train_xgboost
from irrigation.field.shap_analysis import run_shap_analysis

CLASS_NAMES_MAP = {1: "Flood", 2: "Sprinkler", 3: "Drip"}


def _build_label_encoder(y_train: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    """Build maps between original class IDs and zero-based contiguous IDs."""
    train_labels = sorted(np.unique(y_train).tolist())
    label_to_idx = {orig: idx for idx, orig in enumerate(train_labels)}
    idx_to_label = {idx: orig for orig, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/field_level")
    parser.add_argument("--drop_drip", action="store_true", default=True)
    parser.add_argument("--keep_drip", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    drop_drip = args.drop_drip and not args.keep_drip
    if drop_drip:
        train_df = train_df[train_df["label"] != 3]
        test_df = test_df[test_df["label"] != 3]

    print(f"Train: {len(train_df)} fields")
    print(f"Test: {len(test_df)} fields")

    meta_cols = ["state", "tile_id", "field_id", "label"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train = np.nan_to_num(
        train_df[feature_cols].values.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    X_test = np.nan_to_num(
        test_df[feature_cols].values.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    label_to_idx, idx_to_label = _build_label_encoder(y_train)
    print(f"Label remap (original -> encoded): {label_to_idx}")

    # Keep only test samples that are represented in train labels.
    seen_mask = np.isin(y_test, list(label_to_idx.keys()))
    if not np.all(seen_mask):
        unseen = sorted(np.unique(y_test[~seen_mask]).tolist())
        print(f"Warning: dropping {np.sum(~seen_mask)} test samples with unseen labels: {unseen}")
        X_test = X_test[seen_mask]
        y_test = y_test[seen_mask]

    y_train_enc = np.array([label_to_idx[int(v)] for v in y_train], dtype=np.int64)
    y_test_enc = np.array([label_to_idx[int(v)] for v in y_test], dtype=np.int64)

    model = train_xgboost(X_train, y_train_enc, X_test, y_test_enc)
    y_pred_raw = model.predict(X_test)
    if isinstance(y_pred_raw, np.ndarray) and y_pred_raw.ndim == 2:
        # For softprob objectives, predict may return class probabilities.
        y_pred_enc = np.argmax(y_pred_raw, axis=1)
    else:
        y_pred_enc = np.asarray(y_pred_raw).astype(np.int64)
    y_pred = np.array([idx_to_label[int(v)] for v in y_pred_enc], dtype=np.int64)

    present_classes = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    class_names = [CLASS_NAMES_MAP.get(c, f"Class_{c}") for c in present_classes]

    report = classification_report(y_test, y_pred, labels=present_classes, target_names=class_names, digits=4)
    print("\n--- Classification Report (OOD Test Set) ---")
    print(report)
    (output_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Field-Level Confusion Matrix (OOD)")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    run_shap_analysis(
        model=model,
        X_test=X_test,
        feature_names=feature_cols,
        class_names=class_names,
        output_dir=output_dir / "shap",
    )

    model.save_model(str(output_dir / "xgboost_model.json"))
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
