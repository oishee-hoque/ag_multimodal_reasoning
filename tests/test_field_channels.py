import numpy as np
from irrigation.data.field_channels import (
    create_boundary_channel,
    create_distance_channel,
    create_field_size_channel,
    create_all_field_channels,
)


def test_boundary_channel_shape():
    label = np.zeros((224, 224), dtype=np.uint8)
    label[50:100, 50:100] = 1  # flood field
    label[120:160, 80:150] = 2  # sprinkler field

    boundary = create_boundary_channel(label)
    assert boundary.shape == (224, 224)
    assert boundary.dtype == np.float32
    # Boundary should be 1 at edges, 0 elsewhere
    assert boundary[75, 75] == 0.0  # interior
    assert boundary.max() == 1.0
    assert boundary[50, 75] == 1.0 or boundary[51, 75] == 1.0  # near edge


def test_distance_channel_shape():
    label = np.zeros((224, 224), dtype=np.uint8)
    label[50:100, 50:100] = 1

    dist = create_distance_channel(label)
    assert dist.shape == (224, 224)
    assert dist.dtype == np.float32
    assert dist.min() >= 0.0
    assert dist.max() <= 1.0
    # Center should have higher distance than edge
    assert dist[75, 75] > dist[51, 51]
    # Background should be 0
    assert dist[0, 0] == 0.0


def test_field_size_channel():
    label = np.zeros((224, 224), dtype=np.uint8)
    label[10:20, 10:20] = 1    # small field: 100 pixels
    label[50:150, 50:150] = 2  # large field: 10000 pixels

    size = create_field_size_channel(label)
    assert size.shape == (224, 224)
    # Large field pixels should have higher value
    assert size[100, 100] > size[15, 15]
    # Background should be 0
    assert size[0, 0] == 0.0


def test_all_field_channels_shape():
    label = np.zeros((224, 224), dtype=np.uint8)
    label[50:100, 50:100] = 1
    label[120:180, 80:160] = 2

    channels = create_all_field_channels(label)
    assert channels.shape == (3, 224, 224)
    assert channels.dtype == np.float32


def test_empty_label():
    """All background — channels should be all zeros."""
    label = np.zeros((224, 224), dtype=np.uint8)
    channels = create_all_field_channels(label)
    assert channels.shape == (3, 224, 224)
    assert channels.max() == 0.0


def test_ignore_class_excluded():
    """Pixels with class 255 (ignore) should not generate boundaries."""
    label = np.zeros((224, 224), dtype=np.uint8)
    label[50:100, 50:100] = 255  # ignore region

    channels = create_all_field_channels(label)
    assert channels.max() == 0.0  # no field channels for ignore class
