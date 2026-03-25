"""Field-level classification pipeline components.

Keep package import lightweight: do not eagerly import optional dependencies
(e.g. `shap`, `xgboost`) from submodules.
"""

__all__ = [
    "classifiers",
    "feature_extraction",
    "field_datamodule",
    "field_dataset",
    "instance_masks",
    "shap_analysis",
]
