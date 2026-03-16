"""Tests for model forward passes."""

import pytest
import torch
import segmentation_models_pytorch as smp

from irrigation.models.factory import create_model, MODEL_REGISTRY


class TestModelForward:
    """Verify forward pass produces correct output shapes for all models."""

    @pytest.mark.parametrize(
        "model_name",
        list(MODEL_REGISTRY.keys()),
    )
    def test_forward_no_pretrained(self, model_name):
        """All models produce (B, num_classes, H, W) without pretrained weights."""
        # Use >3 channels to avoid pretrained weight download (no network)
        model = create_model(model_name, in_channels=5, num_classes=4)
        model.eval()
        x = torch.randn(2, 5, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4, 224, 224)

    @pytest.mark.parametrize("in_channels", [3, 14, 42])
    def test_forward_multichannel(self, in_channels):
        """U-Net works with various channel counts (no pretrained)."""
        # Build directly with encoder_weights=None to avoid download
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=4,
        )
        model.eval()
        x = torch.randn(2, in_channels, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4, 224, 224)

    def test_unknown_model_raises(self):
        """Creating an unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")

    def test_num_classes_respected(self):
        """Output channel count matches num_classes."""
        for nc in [2, 4, 10]:
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=nc,
            )
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            assert out.shape[1] == nc

    def test_factory_creates_different_architectures(self):
        """Factory produces distinct model types."""
        unet = create_model("unet_resnet34", in_channels=5, num_classes=4)
        deeplab = create_model("deeplabv3plus_resnet50", in_channels=5, num_classes=4)
        assert type(unet) != type(deeplab)
