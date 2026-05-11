#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from process_product_oct_nov_p25 import (
    DEFAULT_LAYERS,
    default_layer,
    output_path_for_hour,
    process_hour,
)


def file_timestamp(product: str, path: Path) -> datetime:
    match = re.search(rf"PUNCH_L3_{re.escape(product.upper())}_(\d{{14}})_", path.name)
    if not match:
        raise ValueError(f"Cannot parse {product.upper()} timestamp from {path.name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def discover_files(input_root: Path, pattern: str, *, recursive: bool) -> list[Path]:
    finder = input_root.rglob if recursive else input_root.glob
    files = [path for path in finder(pattern) if path.is_file()]
    return sorted(files)


def group_by_hour(product: str, paths: list[Path]) -> tuple[dict[str, list[Path]], list[Path]]:
    hours: dict[str, list[Path]] = {}
    rejected: list[Path] = []
    for path in paths:
        try:
            timestamp = file_timestamp(product, path)
        except ValueError:
            rejected.append(path)
            continue
        key = timestamp.strftime("%Y%m%d%H")
        hours.setdefault(key, []).append(path)
    return {key: sorted(value) for key, value in sorted(hours.items())}, rejected


def process_hour_job(
    hour_key: str,
    paths: list[Path],
    output_root: Path,
    *,
    product: str,
    layer: str,
    bin_size_deg: float,
) -> tuple[str, int, int, Path | None, float]:
    start = time.perf_counter()
    written = process_hour(
        hour_key,
        paths,
        output_root,
        product=product,
        layer=layer,
        bin_size_deg=bin_size_deg,
    )
    return hour_key, len(paths), 1 if written is not None else 0, written, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a local pile of PUNCH FITS files into one hourly p25 tomography TXT per hour."
    )
    parser.add_argument("--product", required=True, choices=sorted(DEFAULT_LAYERS))
    parser.add_argument("--layer", default="", help="Layer name. Defaults by product.")
    parser.add_argument("--input-root", default=".", help="Folder containing FITS files")
    parser.add_argument(
        "--pattern",
        default="",
        help="Glob pattern relative to input-root. Defaults to PUNCH_L3_PRODUCT_*.fits",
    )
    parser.add_argument("--recursive", action="store_true", help="Search input-root recursively")
    parser.add_argument("--output-root", default="", help="Defaults to outputs/<product>_local_hourly_p25")
    parser.add_argument("--hour-workers", type=int, default=1, help="Number of local hours to process at once")
    parser.add_argument("--bin-size-deg", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true", help="Reprocess hours whose TXT already exists")
    parser.add_argument("--min-files-per-hour", type=int, default=1)
    parser.add_argument(
        "--hour-filter",
        default="",
        help="Optional regex matched against YYYYMMDDHH hour keys, useful for small test runs",
    )
    parser.add_argument("--max-hours", type=int, default=0, help="Optional cap on processed hours")
    args = parser.parse_args()

    product = args.product.upper()
    layer = args.layer or default_layer(product)
    input_root = Path(args.input_root).expanduser()
    pattern = args.pattern or f"PUNCH_L3_{product}_*.fits"
    output_root = Path(args.output_root or (ROOT / "outputs" / f"{product.lower()}_local_hourly_p25")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    hour_filter = re.compile(args.hour_filter) if args.hour_filter else None

    print(
        f"[start] local {product} p25 hourly composites input_root={input_root} "
        f"pattern={pattern} recursive={args.recursive} layer={layer} "
        f"hour_workers={args.hour_workers}",
        flush=True,
    )

    paths = discover_files(input_root, pattern, recursive=args.recursive)
    print(f"[discover] matched_files={len(paths)}", flush=True)
    hours, rejected = group_by_hour(product, paths)
    if rejected:
        print(f"[discover] rejected_unparseable={len(rejected)}", flush=True)

    totals = {"hours_seen": 0, "hours_done": 0, "hours_skipped": 0, "files": 0}
    jobs: list[tuple[str, list[Path]]] = []
    for hour_key, hour_paths in hours.items():
        if hour_filter is not None and hour_filter.search(hour_key) is None:
            continue
        if args.max_hours and totals["hours_done"] >= args.max_hours:
            break
        totals["hours_seen"] += 1
        first_dt = file_timestamp(product, hour_paths[0]).replace(tzinfo=None)
        out_path = output_path_for_hour(output_root, product, layer, first_dt)
        if out_path.exists() and not args.overwrite:
            totals["hours_skipped"] += 1
            print(f"[exists] {out_path}", flush=True)
            continue
        if len(hour_paths) < args.min_files_per_hour:
            totals["hours_skipped"] += 1
            print(f"[skip-hour] {hour_key}: only {len(hour_paths)} local files", flush=True)
            continue
        jobs.append((hour_key, hour_paths))

    if args.max_hours:
        remaining = max(args.max_hours - totals["hours_done"], 0)
        jobs = jobs[:remaining]

    run_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.hour_workers)) as pool:
        futures = {
            pool.submit(
                process_hour_job,
                hour_key,
                hour_paths,
                output_root,
                product=product,
                layer=layer,
                bin_size_deg=args.bin_size_deg,
            ): hour_key
            for hour_key, hour_paths in jobs
        }
        for future in as_completed(futures):
            hour_key = futures[future]
            try:
                _, file_count, done_count, _written, elapsed = future.result()
                totals["files"] += file_count
                if done_count:
                    totals["hours_done"] += 1
                else:
                    totals["hours_skipped"] += 1
                print(f"[hour-done] {hour_key} files={file_count} seconds={elapsed:.1f}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep batch runs moving.
                totals["hours_skipped"] += 1
                print(f"[hour-failed] {hour_key}: {exc}", flush=True)

    print(
        "[done] "
        f"hours_seen={totals['hours_seen']} hours_done={totals['hours_done']} "
        f"hours_skipped={totals['hours_skipped']} files_processed={totals['files']} "
        f"seconds={time.perf_counter() - run_start:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
