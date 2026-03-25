"""Field-level classification pipeline components."""

from irrigation.field.classifiers import FieldMLPClassifier, FeatureGroupAttention, train_xgboost
from irrigation.field.feature_extraction import extract_all_features
from irrigation.field.field_datamodule import FieldDataModule
from irrigation.field.field_dataset import FieldFeatureDataset
from irrigation.field.instance_masks import generate_all_instance_masks, load_irrigation_polygons
from irrigation.field.shap_analysis import run_shap_analysis

__all__ = [
    "FieldDataModule",
    "FieldFeatureDataset",
    "FieldMLPClassifier",
    "FeatureGroupAttention",
    "extract_all_features",
    "generate_all_instance_masks",
    "load_irrigation_polygons",
    "run_shap_analysis",
    "train_xgboost",
]
