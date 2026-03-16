"""
Model registry and factory.

Uses segmentation-models-pytorch (smp) as the backbone library.
Supports swappable encoders and configurable input channels.

For pretrained encoders (ImageNet), input must be 3 channels.
For >3 channels, use from-scratch encoders or adapt the first conv layer.
"""

import segmentation_models_pytorch as smp


MODEL_REGISTRY = {}


def register_model(name: str):
    """Decorator to register a model constructor."""

    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


@register_model("unet_resnet34")
def unet_resnet34(in_channels: int = 3, num_classes: int = 4, **kwargs):
    """U-Net with pretrained ResNet-34 encoder. Best for 3-channel (RGB) input."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if in_channels == 3 else None,
        in_channels=in_channels,
        classes=num_classes,
    )


@register_model("unet_resnet50")
def unet_resnet50(in_channels: int = 3, num_classes: int = 4, **kwargs):
    """U-Net with pretrained ResNet-50 encoder."""
    return smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet" if in_channels == 3 else None,
        in_channels=in_channels,
        classes=num_classes,
    )


@register_model("unet_efficientnet_b3")
def unet_efficientnet_b3(in_channels: int = 3, num_classes: int = 4, **kwargs):
    """U-Net with EfficientNet-B3 encoder."""
    return smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet" if in_channels == 3 else None,
        in_channels=in_channels,
        classes=num_classes,
    )


@register_model("deeplabv3plus_resnet50")
def deeplabv3plus_resnet50(in_channels: int = 3, num_classes: int = 4, **kwargs):
    """DeepLabV3+ with ResNet-50 encoder."""
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet" if in_channels == 3 else None,
        in_channels=in_channels,
        classes=num_classes,
    )


def create_model(name: str, **kwargs):
    """
    Create a model by registered name.

    Args:
        name: registered model name (e.g., "unet_resnet34")
        **kwargs: in_channels, num_classes, etc.

    Returns:
        nn.Module
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
