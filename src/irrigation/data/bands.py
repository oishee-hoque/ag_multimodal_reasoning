"""
Band configuration system.

Each band group defines which bands to extract from which seasons.
The dataset reads the active band group from the Hydra config and
extracts only those channels.
"""

from dataclasses import dataclass


@dataclass
class BandConfig:
    """Defines which channels to load from the GeoTIFFs."""

    name: str
    band_indices: list[int]  # which of the 14 bands per season
    seasons: list[str]  # which season files to load
    num_channels: int  # total input channels to the model
    pretrained_compatible: bool  # can use ImageNet pretrained weights


# Band group definitions
BAND_GROUPS = {
    "rgb_s4": BandConfig(
        name="rgb_s4",
        band_indices=[0, 1, 2],  # B02, B03, B04
        seasons=["s4"],  # peak summer only
        num_channels=3,
        pretrained_compatible=True,
    ),
    "rgb_s5": BandConfig(
        name="rgb_s5",
        band_indices=[0, 1, 2],
        seasons=["s5"],
        num_channels=3,
        pretrained_compatible=True,
    ),
    "spectral_s4": BandConfig(
        name="spectral_s4",
        band_indices=list(range(14)),  # all 14 bands
        seasons=["s4"],
        num_channels=14,
        pretrained_compatible=False,
    ),
    "temporal": BandConfig(
        name="temporal",
        band_indices=list(range(14)),
        seasons=["s3", "s4", "s5"],
        num_channels=42,  # 14 × 3
        pretrained_compatible=False,
    ),
}


def get_band_config(name: str) -> BandConfig:
    """Retrieve a band configuration by name."""
    if name not in BAND_GROUPS:
        raise ValueError(f"Unknown band group: {name}. Available: {list(BAND_GROUPS.keys())}")
    return BAND_GROUPS[name]
