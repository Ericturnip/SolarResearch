#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import glob
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from punch_pipeline_v4.adapters.registry import get_adapter
from punch_pipeline_v4.processing.binning import median_bin_layer_frame
from punch_pipeline_v4.processing.composite import composite_binned_maps
from punch_pipeline_v4.processing.units import apply_s10_filter
from punch_pipeline_v4.writers.ascii import write_tomography_txt


def main():
    ap = argparse.ArgumentParser(description="Build one-hour binned composite from PUNCH FITS files")
    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--product", required=True, choices=["CIM", "CTM", "PIM", "PTM", "PAM", "CAM"])
    ap.add_argument("--layer", required=True)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "composite"))
    ap.add_argument("--bin-size-deg", type=float, default=1.0)
    ap.add_argument("--composite-method", default="nanmedian", choices=["nanmedian", "nanmean", "nanmin", "p30", "percentile"])
    ap.add_argument("--percentile", type=float, default=30.0)
    ap.add_argument("--convert-to-s10", action="store_true")
    ap.add_argument("--drop-zero-bins", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--drop-zero-before-stat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--positive-only", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    adapter = get_adapter(args.product)
    binned_maps = []
    print(f"[INPUT] selected {len(files)} files")
    for fp in files:
        special = adapter.make_binned_map(
            fp,
            args.layer,
            bin_size_deg=args.bin_size_deg,
            convert_to_s10=args.convert_to_s10,
        )

        if special is not None:
            bmap, diag = special
            print(
                f"  {Path(fp).name}: DATE-OBS={bmap.timestamp.isot}, "
                f"layer={bmap.layer_name}, unit={bmap.unit}"
            )
        else:
            frame = adapter.load_layer(fp, args.layer)
            print(
                f"  {Path(fp).name}: DATE-OBS={frame.timestamp.isot}, "
                f"layer={frame.layer_name}, unit={frame.native_unit}"
            )

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

        binned_maps.append(bmap)

    result = composite_binned_maps(
        binned_maps,
        method=args.composite_method,
        percentile=args.percentile,
        drop_zero_before_stat=args.drop_zero_before_stat,
    )

    if args.convert_to_s10:
        result.binned_map.unit = "S10"

    start = binned_maps[0].timestamp.strftime("%Y%m%d%H%M%S")
    out_dir = Path(args.output_dir).expanduser()
    out_name = f"PUNCH_L3_{args.product}_{start}_{args.layer}_{args.composite_method}_COMPOSITE.txt"
    report = write_tomography_txt(
        result.binned_map,
        out_dir / out_name,
        values_are_s10=args.convert_to_s10,
        drop_zero_bins=args.drop_zero_bins,
        positive_only=args.positive_only,
    )

    print("[COMPOSITE DIAGNOSTICS]")
    for k, v in result.diagnostics.items():
        print(f"{k}: {v}")
    print("[WRITE]")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
