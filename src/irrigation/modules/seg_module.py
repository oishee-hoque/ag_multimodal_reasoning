"""
Lightning Module for semantic segmentation.

Handles:
- Loss computation (CE with ignore_index, optional class weights)
- Metric tracking (per-class IoU, mIoU, confusion matrix)
- Optimizer and scheduler configuration
- W&B logging of metrics, sample predictions, confusion matrices
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.segmentation import MeanIoU
from torchmetrics import ConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from irrigation.models.factory import create_model


class SegmentationModule(pl.LightningModule):
    """
    Lightning module wrapping any segmentation model.

    Args:
        model_name: registered model name
        in_channels: input channels (from band config)
        num_classes: number of output classes (4: bg, flood, sprinkler, drip)
        class_weights: optional tensor of class weights for imbalanced data
        ignore_index: label value to ignore in loss (255)
        lr: learning rate
        weight_decay: AdamW weight decay
        scheduler: "cosine" or "plateau"
        max_epochs: for cosine scheduler
        class_names: list of class name strings for logging
    """

    def __init__(
        self,
        model_name: str = "unet_resnet34",
        in_channels: int = 3,
        num_classes: int = 4,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        max_epochs: int = 100,
        class_names: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = create_model(
            model_name, in_channels=in_channels, num_classes=num_classes
        )

        # Loss
        weight = torch.tensor(class_weights) if class_weights else None
        self.register_buffer("class_weight", weight)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.class_names = class_names or [
            "background",
            "flood",
            "sprinkler",
            "drip",
        ]

        # Metrics — separate instances for val and test
        metric_kwargs = dict(num_classes=num_classes, per_class=True, input_format="index")
        self.val_iou = MeanIoU(**metric_kwargs)
        self.test_iou = MeanIoU(**metric_kwargs)
        self.val_confusion = ConfusionMatrix(
            task="multiclass", num_classes=num_classes, ignore_index=ignore_index
        )
        self.test_confusion = ConfusionMatrix(
            task="multiclass", num_classes=num_classes, ignore_index=ignore_index
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            labels,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])  # (B, C, H, W)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        logits = self(batch["image"])
        loss = self._compute_loss(logits, batch["label"])
        preds = logits.argmax(dim=1)

        # Update metrics (filter out ignore pixels)
        mask = batch["label"] != self.ignore_index
        if mask.any():
            self.val_iou.update(preds[mask], batch["label"][mask])
            self.val_confusion.update(preds, batch["label"])

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        # Log sample predictions for first batch
        if batch_idx == 0 and self.logger:
            self._log_predictions(batch["image"], batch["label"], preds, "val")

    def on_validation_epoch_end(self):
        iou = self.val_iou.compute()
        self.log("val/mIoU", iou.mean(), prog_bar=True)
        for i, name in enumerate(self.class_names):
            self.log(f"val/IoU_{name}", iou[i])

        # Log confusion matrix
        if self.logger and hasattr(self.logger.experiment, "log"):
            cm = self.val_confusion.compute().cpu().numpy()
            self._log_confusion_matrix(cm, "val")

        self.val_iou.reset()
        self.val_confusion.reset()

    def test_step(self, batch: dict, batch_idx: int):
        logits = self(batch["image"])
        loss = self._compute_loss(logits, batch["label"])
        preds = logits.argmax(dim=1)

        mask = batch["label"] != self.ignore_index
        if mask.any():
            self.test_iou.update(preds[mask], batch["label"][mask])
            self.test_confusion.update(preds, batch["label"])

        self.log("test/loss", loss, on_epoch=True)

    def on_test_epoch_end(self):
        iou = self.test_iou.compute()
        self.log("test/mIoU", iou.mean())
        for i, name in enumerate(self.class_names):
            self.log(f"test/IoU_{name}", iou[i])

        cm = self.test_confusion.compute().cpu().numpy()
        if self.logger:
            self._log_confusion_matrix(cm, "test")

        self.test_iou.reset()
        self.test_confusion.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mIoU",
                },
            }
        else:
            return optimizer

    def _log_predictions(
        self, images, labels, preds, stage: str, max_samples: int = 4
    ):
        """Log sample prediction overlays to W&B."""
        try:
            import wandb
        except ImportError:
            return

        n = min(max_samples, images.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        class_colors = np.array([
            [0, 0, 0],        # background - black
            [0, 0, 255],      # flood - blue
            [0, 255, 0],      # sprinkler - green
            [255, 0, 0],      # drip - red
        ])

        for i in range(n):
            img = images[i].cpu().numpy()
            # Use first 3 channels as RGB composite (or just first 3 if >3 channels)
            rgb = img[:3].transpose(1, 2, 0)  # (H, W, 3)
            # Normalize for display
            rgb_min, rgb_max = rgb.min(), rgb.max()
            if rgb_max > rgb_min:
                rgb = (rgb - rgb_min) / (rgb_max - rgb_min)

            lbl = labels[i].cpu().numpy()
            pred = preds[i].cpu().numpy()

            # Create colored label maps (ignore_index=255 → gray)
            lbl_safe = np.where(lbl == 255, 4, np.clip(lbl, 0, 3))
            ignore_color = np.array([[128, 128, 128]])  # gray for ignored pixels
            all_colors = np.concatenate([class_colors, ignore_color], axis=0)
            lbl_rgb = all_colors[lbl_safe] / 255.0
            pred_rgb = class_colors[np.clip(pred, 0, 3)] / 255.0

            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title("Input (RGB)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(lbl_rgb)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

        plt.tight_layout()
        self.logger.experiment.log({f"{stage}/predictions": wandb.Image(fig)})
        plt.close(fig)

    def _log_confusion_matrix(self, cm: np.ndarray, stage: str):
        """Log confusion matrix as W&B plot."""
        try:
            import wandb
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{stage} Confusion Matrix")
        plt.tight_layout()
        self.logger.experiment.log({f"{stage}/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
