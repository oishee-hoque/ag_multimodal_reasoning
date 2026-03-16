"""Tests for metric computation."""

import pytest
import torch
from torchmetrics.segmentation import MeanIoU
from torchmetrics import ConfusionMatrix


class TestMetrics:
    """Verify IoU and confusion matrix computations."""

    def test_perfect_iou(self):
        """Perfect predictions yield IoU of 1.0."""
        metric = MeanIoU(num_classes=4, per_class=True, input_format="index")
        # MeanIoU expects at least 2D: (B, ...) or (H, W)
        preds = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        target = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        metric.update(preds, target)
        iou = metric.compute()
        assert torch.allclose(iou, torch.ones(4))

    def test_zero_iou_for_wrong_class(self):
        """Completely wrong predictions for a class yield IoU of 0."""
        metric = MeanIoU(num_classes=4, per_class=True, input_format="index")
        # All predicted as 0, but targets are 1
        preds = torch.zeros(1, 100, dtype=torch.long)
        target = torch.ones(1, 100, dtype=torch.long)
        metric.update(preds, target)
        iou = metric.compute()
        # Class 1 IoU should be 0 (no true positives)
        assert iou[1] == 0.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix has shape (num_classes, num_classes)."""
        metric = ConfusionMatrix(task="multiclass", num_classes=4)
        preds = torch.tensor([0, 1, 2, 3, 0, 1])
        target = torch.tensor([0, 1, 2, 3, 1, 0])
        metric.update(preds, target)
        cm = metric.compute()
        assert cm.shape == (4, 4)

    def test_confusion_matrix_ignore_index(self):
        """Confusion matrix correctly ignores pixels with ignore_index."""
        metric = ConfusionMatrix(
            task="multiclass", num_classes=4, ignore_index=255
        )
        preds = torch.tensor([0, 1, 2, 0])
        target = torch.tensor([0, 1, 255, 0])
        metric.update(preds, target)
        cm = metric.compute()
        # Total non-ignored samples = 3
        assert cm.sum() == 3

    def test_miou_computation(self):
        """Mean IoU is the average of per-class IoUs."""
        metric = MeanIoU(num_classes=4, per_class=True, input_format="index")
        preds = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
        target = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
        metric.update(preds, target)
        iou = metric.compute()
        miou = iou.mean()
        assert miou == 1.0
