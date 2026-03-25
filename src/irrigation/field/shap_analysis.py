"""SHAP feature importance analysis for field-level classifiers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap


def run_shap_analysis(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    output_dir: str | Path,
    max_samples: int = 1000,
):
    """Run and persist SHAP analyses for a trained tree model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(X_test) > max_samples:
        idx = np.random.RandomState(42).choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_per_class: list[np.ndarray] | None = None
    if isinstance(shap_values, list):
        shap_per_class = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Some SHAP/XGBoost versions return (n_samples, n_features, n_classes).
        shap_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    shap_for_summary = shap_per_class if shap_per_class is not None else shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_for_summary,
        X_sample,
        feature_names=feature_names,
        class_names=class_names,
        show=False,
        max_display=30,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_for_summary,
        X_sample,
        feature_names=feature_names,
        class_names=class_names,
        plot_type="bar",
        show=False,
        max_display=30,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    if shap_per_class is not None:
        iterable = enumerate(class_names[: len(shap_per_class)])
    else:
        iterable = []
    for cls_idx, cls_name in iterable:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_per_class[cls_idx],
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_{cls_name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()

    import pandas as pd

    if shap_per_class is not None:
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_per_class], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(output_dir / "top_features.csv", index=False)

    print(f"SHAP analysis saved to {output_dir}")
    print("\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))
    return shap_values, importance_df
