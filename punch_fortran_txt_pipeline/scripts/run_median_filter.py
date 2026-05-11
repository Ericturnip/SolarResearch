#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from punch_pipeline_v4.adapters.registry import get_adapter
from punch_pipeline_v4.processing.binning import median_bin_layer_frame
from punch_pipeline_v4.processing.units import native_to_s10, apply_s10_filter
from punch_pipeline_v4.writers.ascii import write_tomography_txt


def main():
    ap = argparse.ArgumentParser(description="Median-bin one PUNCH image to 1x1 degree TXT")
    ap.add_argument("--input", required=True)
    ap.add_argument("--product", required=True, choices=["CIM", "CTM", "PIM", "PTM", "PAM"])
    ap.add_argument("--layer", required=True)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "median"))
    ap.add_argument("--bin-size-deg", type=float, default=1.0)
    ap.add_argument("--convert-to-s10", action="store_true", help="Convert native brightness to S10 before writing")
    ap.add_argument("--drop-zero-bins", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--positive-only", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    adapter = get_adapter(args.product)

    special = adapter.make_binned_map(
        args.input,
        args.layer,
        bin_size_deg=args.bin_size_deg,
        convert_to_s10=args.convert_to_s10,
    )

    if special is not None:
        bmap, diag = special
    else:
        frame = adapter.load_layer(args.input, args.layer)
        bmap, diag = median_bin_layer_frame(
            frame,
            bin_size_deg=args.bin_size_deg,
            convert_to_s10=args.convert_to_s10,
        )

    if args.convert_to_s10:
        bmap.values, _ = apply_s10_filter(
            bmap.values,
            drop_zero=args.drop_zero_bins,
            positive_only=args.positive_only,
        )
        bmap.unit = "S10"

    out_dir = Path(args.output_dir).expanduser()
    out_name = f"PUNCH_L3_{args.product}_{bmap.timestamp.strftime('%Y%m%d%H%M%S')}_{bmap.layer_name}_BIN.txt"
    report = write_tomography_txt(
        bmap,
        out_dir / out_name,
        values_are_s10=args.convert_to_s10,
        drop_zero_bins=args.drop_zero_bins,
        positive_only=args.positive_only,
    )

    print("[DIAGNOSTICS]")
    for k, v in diag.items():
        print(f"{k}: {v}")
    print("[WRITE]")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
