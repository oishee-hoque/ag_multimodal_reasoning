"""Step 5: Compare pixel-level and field-level results from text reports."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _extract_weighted_f1(report_text: str) -> float | None:
    for line in report_text.splitlines():
        if "weighted avg" in line:
            parts = re.split(r"\s+", line.strip())
            # weighted avg precision recall f1 support
            if len(parts) >= 6:
                try:
                    return float(parts[-2])
                except ValueError:
                    return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field_report", required=True, type=str)
    parser.add_argument("--pixel_report", required=True, type=str)
    parser.add_argument("--output_csv", type=str, default="results/approach_comparison.csv")
    args = parser.parse_args()

    field_text = Path(args.field_report).read_text()
    pixel_text = Path(args.pixel_report).read_text()

    field_f1 = _extract_weighted_f1(field_text)
    pixel_f1 = _extract_weighted_f1(pixel_text)

    df = pd.DataFrame(
        [
            {"approach": "field_level", "weighted_f1": field_f1},
            {"approach": "pixel_level", "weighted_f1": pixel_f1},
            {"approach": "delta_field_minus_pixel", "weighted_f1": None if field_f1 is None or pixel_f1 is None else field_f1 - pixel_f1},
        ]
    )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df)
    print(f"Saved comparison to {out}")


if __name__ == "__main__":
    main()
