"""Tests for augmentation transforms."""

import numpy as np
import pytest

from irrigation.data.transforms import get_train_transforms, get_val_transforms


class TestTransforms:
    """Verify augmentations preserve label consistency."""

    def test_train_transforms_preserve_shape(self):
        """Train transforms maintain image and mask shape."""
        transform = get_train_transforms()
        image = np.random.rand(224, 224, 3).astype(np.float32)
        mask = np.random.choice([0, 1, 2, 3], size=(224, 224)).astype(np.uint8)

        result = transform(image=image, mask=mask)
        assert result["image"].shape == (224, 224, 3)
        assert result["mask"].shape == (224, 224)

    def test_val_transforms_identity(self):
        """Val transforms don't change the data."""
        transform = get_val_transforms()
        image = np.random.rand(224, 224, 3).astype(np.float32)
        mask = np.random.choice([0, 1, 2, 3], size=(224, 224)).astype(np.uint8)

        result = transform(image=image, mask=mask)
        np.testing.assert_array_equal(result["image"], image)
        np.testing.assert_array_equal(result["mask"], mask)

    def test_train_transforms_preserve_label_values(self):
        """Geometric transforms don't create new label values."""
        transform = get_train_transforms()
        image = np.random.rand(224, 224, 3).astype(np.float32)
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[50:150, 50:150] = 2
        mask[0:10, 0:10] = 255

        original_values = set(np.unique(mask))

        # Run many times to test various random augmentations
        for _ in range(20):
            result = transform(image=image, mask=mask)
            result_values = set(np.unique(result["mask"]))
            assert result_values.issubset(original_values)

    def test_train_transforms_multichannel(self):
        """Transforms work with >3 channel images."""
        transform = get_train_transforms()
        image = np.random.rand(224, 224, 42).astype(np.float32)
        mask = np.random.choice([0, 1, 2, 3], size=(224, 224)).astype(np.uint8)

        result = transform(image=image, mask=mask)
        assert result["image"].shape == (224, 224, 42)
        assert result["mask"].shape == (224, 224)
