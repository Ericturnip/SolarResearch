#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import glob
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from punch_pipeline_v4.processing.second_pass import SecondPassConfig, run_second_pass_on_txt


def main():
    ap = argparse.ArgumentParser(description="Run second-pass cleaning on first-pass TXT files")
    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "second_pass"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--coverage-min", type=float, default=None)
    ap.add_argument("--drop-zero-bins", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = SecondPassConfig(threshold=args.threshold, coverage_min=args.coverage_min, drop_zero_bins=args.drop_zero_bins)
    for fp in files:
        out = out_dir / (Path(fp).stem + "_SECONDPASS.txt")
        print(f"[SECOND PASS] {fp} -> {out}")
        report = run_second_pass_on_txt(fp, out, config)
        print(report)


if __name__ == "__main__":
    main()
