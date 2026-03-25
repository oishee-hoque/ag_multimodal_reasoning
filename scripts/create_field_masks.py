"""Step 1: Generate per-tile field instance masks from .gdb polygons."""

from __future__ import annotations

import argparse
from pathlib import Path

from irrigation.field.instance_masks import generate_all_instance_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", type=str, required=True, help="Path to state data dir")
    parser.add_argument("--gdb_path", type=str, required=True, help="Path to .gdb file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir (default: state_path/field_data)",
    )
    parser.add_argument("--min_field_pixels", type=int, default=20)
    args = parser.parse_args()

    state_path = Path(args.state_path)
    output_dir = Path(args.output_dir) if args.output_dir else state_path / "field_data"

    generate_all_instance_masks(
        state_path=state_path,
        gdb_path=Path(args.gdb_path),
        output_dir=output_dir,
        min_field_pixels=args.min_field_pixels,
    )


if __name__ == "__main__":
    main()
