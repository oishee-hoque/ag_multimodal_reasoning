"""Field-level classifiers: XGBoost baseline and MLP with feature-group attention."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
):
    """Train an XGBoost classifier on field features."""
    import xgboost as xgb

    n_classes = int(np.unique(y_train).shape[0])
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes for classification, got {n_classes}")

    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    # XGBoost requires num_class for multi-class objectives.
    objective = str(default_params.get("objective", ""))
    if objective.startswith("multi:") and "num_class" not in default_params:
        default_params["num_class"] = n_classes

    model = xgb.XGBClassifier(**default_params)
    eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
    return model


class FeatureGroupAttention(nn.Module):
    """Attention block over feature groups (spectral, temporal, shape)."""

    def __init__(
        self,
        group_dims: list[tuple[str, int]],
        hidden_dim: int = 64,
        knowledge_prior: torch.Tensor | None = None,
    ):
        super().__init__()
        self.group_names = [name for name, _ in group_dims]
        self.group_sizes = [size for _, size in group_dims]
        self.num_groups = len(group_dims)

        self.group_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(size, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _, size in group_dims
            ]
        )
        self.attention_query = nn.Linear(hidden_dim, 1)
        if knowledge_prior is not None:
            self.register_buffer("knowledge_prior", knowledge_prior)
        else:
            self.knowledge_prior = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = torch.split(x, self.group_sizes, dim=1)
        projected = [proj(group) for proj, group in zip(self.group_projections, groups)]
        stacked = torch.stack(projected, dim=1)
        logits = self.attention_query(stacked).squeeze(-1)
        if self.knowledge_prior is not None:
            logits = logits + self.knowledge_prior.unsqueeze(0)
        weights = F.softmax(logits, dim=1)
        output = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return output, weights


class FieldMLPClassifier(nn.Module):
    """MLP classifier with feature-group attention front-end."""

    def __init__(
        self,
        feature_group_dims: list[tuple[str, int]],
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
        knowledge_prior: torch.Tensor | None = None,
    ):
        super().__init__()
        attn_dim = 64
        self.attention = FeatureGroupAttention(
            feature_group_dims,
            hidden_dim=attn_dim,
            knowledge_prior=knowledge_prior,
        )
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation, attention_weights = self.attention(x)
        logits = self.classifier(representation)
        return logits, attention_weights
