#!/usr/bin/env python3
"""Download PUNCH L3 files and write one fixed-width TXT map per hour.

This is the main batch script. It is deliberately forgiving: missing NASA days,
bad FITS files, and failed hours are logged, then the run moves on.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from punch_pipeline_v4.adapters.registry import get_adapter
from punch_pipeline_v4.models import BinnedMap
from punch_pipeline_v4.writers.ascii import write_tomography_txt

from process_ctm_oct_nov_p25 import (
    CachedGrid,
    RemoteFile,
    daterange,
    download_hour,
    fetch_text,
    group_by_hour,
    median_bin_fast,
    p25_composite,
    parse_ymd,
    prepare_grid,
    wcs_cache_key,
)


DEFAULT_BASE_ROOT = "https://umbra.nascom.nasa.gov/punch/3"
DEFAULT_LAYERS = {
    "CIM": "brightness",
    "CTM": "brightness",
    "PAM": "Polar_pB",
    "PIM": "Polar_pB",
    "PTM": "Polar_pB",
}


def safe_layer_name(layer: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", layer).strip("_")


def default_layer(product: str) -> str:
    product = product.upper()
    if product not in DEFAULT_LAYERS:
        raise ValueError(f"No default layer for {product}. Known: {sorted(DEFAULT_LAYERS)}")
    return DEFAULT_LAYERS[product]


def default_base_url(product: str) -> str:
    return f"{DEFAULT_BASE_ROOT}/{product.upper()}"


def remote_timestamp(product: str, name: str) -> datetime:
    match = re.search(rf"PUNCH_L3_{re.escape(product.upper())}_(\d{{14}})_", name)
    if not match:
        raise ValueError(f"Cannot parse {product.upper()} timestamp from {name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def list_day_files(
    day: date,
    *,
    product: str,
    base_url: str,
    timeout: float,
    retries: int,
) -> list[RemoteFile]:
    product = product.upper()
    url = f"{base_url.rstrip('/')}/{day.year:04d}/{day.month:02d}/{day.day:02d}/"
    html = fetch_text(url, timeout=timeout, retries=retries)
    if html is None:
        return []

    pattern = rf'href="(PUNCH_L3_{re.escape(product)}_\d{{14}}_[^"]*\.fits)"'
    names = sorted(set(re.findall(pattern, html)))
    out: list[RemoteFile] = []
    for name in names:
        try:
            out.append(
                RemoteFile(
                    name=name,
                    timestamp=remote_timestamp(product, name),
                    url=urljoin(url, name),
                )
            )
        except ValueError:
            continue
    return out


def binned_map_from_file(
    path: Path,
    *,
    product: str,
    layer: str,
    adapter,
    grid_cache: dict[str, CachedGrid],
    bin_size_deg: float,
) -> BinnedMap:
    special = adapter.make_binned_map(
        path,
        layer,
        bin_size_deg=bin_size_deg,
        convert_to_s10=True,
    )
    if special is not None:
        bmap, _diag = special
        return bmap

    frame = adapter.load_layer(path, layer)
    key = wcs_cache_key(frame)
    if key not in grid_cache:
        grid_cache[key] = prepare_grid(frame, bin_size_deg)
        print(
            f"[grid-cache] cached {product.upper()} grid #{len(grid_cache)} "
            f"shape={grid_cache[key].shape}",
            flush=True,
        )
    return median_bin_fast(frame, grid_cache[key])


def output_path_for_hour(output_root: Path, product: str, layer: str, timestamp: datetime) -> Path:
    product = product.upper()
    day_dir = output_root / f"{timestamp.year:04d}" / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
    return day_dir / (
        f"PUNCH_L3_{product}_{timestamp.strftime('%Y%m%d%H')}_"
        f"{safe_layer_name(layer)}_p25_COMPOSITE.txt"
    )


def process_hour(
    hour_key: str,
    paths: list[Path],
    output_root: Path,
    *,
    product: str,
    layer: str,
    bin_size_deg: float,
) -> Path | None:
    if not paths:
        print(f"[skip-hour] {hour_key}: no downloaded FITS", flush=True)
        return None

    product = product.upper()
    adapter = get_adapter(product)
    bmaps: list[BinnedMap] = []
    grid_cache: dict[str, CachedGrid] = {}
    for path in paths:
        try:
            bmaps.append(
                binned_map_from_file(
                    path,
                    product=product,
                    layer=layer,
                    adapter=adapter,
                    grid_cache=grid_cache,
                    bin_size_deg=bin_size_deg,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad frame should not cost the whole hour.
            print(f"[process-file-failed] {path.name}: {exc}", flush=True)

    if not bmaps:
        print(f"[skip-hour] {hour_key}: no processable FITS", flush=True)
        return None

    comp = p25_composite(bmaps)
    dt = bmaps[0].timestamp.to_datetime()
    out_path = output_path_for_hour(output_root, product, comp.layer_name or layer, dt)
    report = write_tomography_txt(
        comp,
        out_path,
        values_are_s10=False,
        drop_zero_bins=True,
    )
    print(
        f"[wrote] {out_path} inputs={len(bmaps)} rows={report['rows_written']} "
        f"removed={report['removed_rows']} "
        f"fixed_width_overflow={report.get('removed_fixed_width_overflow', 0)}",
        flush=True,
    )
    return out_path


def process_hour_job(
    hour_key: str,
    hour_files: list[RemoteFile],
    output_root: Path,
    cache_root: Path,
    *,
    product: str,
    layer: str,
    download_workers: int,
    timeout: float,
    retries: int,
    bin_size_deg: float,
    keep_fits: bool,
) -> tuple[str, int, int, Path | None, float]:
    first_dt = hour_files[0].timestamp
    fits_dir = cache_root / product.upper() / first_dt.strftime("%Y/%m/%d/%H")
    start = time.perf_counter()
    paths = download_hour(
        hour_files,
        fits_dir,
        workers=download_workers,
        timeout=timeout,
        retries=retries,
    )
    written = process_hour(
        hour_key,
        paths,
        output_root,
        product=product,
        layer=layer,
        bin_size_deg=bin_size_deg,
    )
    if not keep_fits and fits_dir.exists():
        shutil.rmtree(fits_dir, ignore_errors=True)
    return hour_key, len(paths), 1 if written is not None else 0, written, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a PUNCH product from NASA, bin each frame to the sky grid, "
            "and write one hourly p25-nearest-real-sample TXT map."
        )
    )
    parser.add_argument("--product", required=True, choices=sorted(DEFAULT_LAYERS))
    parser.add_argument("--layer", default="", help="Layer to read. If omitted, the product default is used.")
    parser.add_argument("--start-date", default="2025-10-01", help="First UTC day to process, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2025-11-30", help="Last UTC day to process, YYYY-MM-DD.")
    parser.add_argument("--base-url", default="", help="Override the NASA product URL.")
    parser.add_argument("--output-root", default="", help="Folder for TXT output. Defaults to outputs/<product>_hourly_p25.")
    parser.add_argument("--cache-root", default=str(ROOT / "outputs" / "_download_cache"), help="Temporary FITS download folder.")
    parser.add_argument("--download-workers", type=int, default=8, help="Parallel FITS downloads inside each hour.")
    parser.add_argument("--hour-workers", type=int, default=1, help="Hours to download and process at the same time.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Network timeout in seconds.")
    parser.add_argument("--retries", type=int, default=4, help="Download/listing attempts before giving up on a file.")
    parser.add_argument("--bin-size-deg", type=float, default=1.0, help="Sky-bin size in degrees.")
    parser.add_argument("--keep-fits", action="store_true", help="Keep downloaded FITS files after each hour finishes.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild TXT files that already exist.")
    parser.add_argument("--min-files-per-hour", type=int, default=1, help="Skip hours with fewer FITS files than this.")
    parser.add_argument(
        "--hour-filter",
        default="",
        help="Regex matched against YYYYMMDDHH, useful for a tiny test run.",
    )
    parser.add_argument("--max-hours", type=int, default=0, help="Stop after this many written hours; 0 means no cap.")
    args = parser.parse_args()

    product = args.product.upper()
    layer = args.layer or default_layer(product)
    start = parse_ymd(args.start_date)
    end = parse_ymd(args.end_date)
    base_url = args.base_url or default_base_url(product)
    output_root = Path(args.output_root or (ROOT / "outputs" / f"{product.lower()}_hourly_p25")).expanduser()
    cache_root = Path(args.cache_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[start] {product} p25 hourly composites {start}..{end} layer={layer} "
        f"hour_workers={args.hour_workers} download_workers={args.download_workers} "
        f"keep_fits={args.keep_fits}",
        flush=True,
    )
    totals = {"days": 0, "hours_seen": 0, "hours_done": 0, "hours_skipped": 0, "files": 0}
    run_start = time.perf_counter()
    hour_filter = re.compile(args.hour_filter) if args.hour_filter else None

    for day in daterange(start, end):
        totals["days"] += 1
        day_start = time.perf_counter()
        print(f"[day] {day.isoformat()}", flush=True)
        remote_files = list_day_files(
            day,
            product=product,
            base_url=base_url,
            timeout=args.timeout,
            retries=args.retries,
        )
        if not remote_files:
            print(f"[day-skip] {day.isoformat()}: no {product} files found", flush=True)
            continue

        hour_jobs = []
        for hour_key, hour_files in group_by_hour(remote_files).items():
            if hour_filter is not None and hour_filter.search(hour_key) is None:
                continue
            if args.max_hours and totals["hours_done"] >= args.max_hours:
                break
            totals["hours_seen"] += 1
            first_dt = hour_files[0].timestamp
            out_path = output_path_for_hour(output_root, product, layer, first_dt)
            if out_path.exists() and not args.overwrite:
                totals["hours_skipped"] += 1
                print(f"[exists] {out_path}", flush=True)
                continue
            if len(hour_files) < args.min_files_per_hour:
                totals["hours_skipped"] += 1
                print(f"[skip-hour] {hour_key}: only {len(hour_files)} remote files", flush=True)
                continue
            hour_jobs.append((hour_key, hour_files))

        if hour_jobs:
            if args.max_hours:
                remaining = max(args.max_hours - totals["hours_done"], 0)
                hour_jobs = hour_jobs[:remaining]
            with ThreadPoolExecutor(max_workers=max(1, args.hour_workers)) as pool:
                futures = {
                    pool.submit(
                        process_hour_job,
                        hour_key,
                        hour_files,
                        output_root,
                        cache_root,
                        product=product,
                        layer=layer,
                        download_workers=args.download_workers,
                        timeout=args.timeout,
                        retries=args.retries,
                        bin_size_deg=args.bin_size_deg,
                        keep_fits=args.keep_fits,
                    ): hour_key
                    for hour_key, hour_files in hour_jobs
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
                    except Exception as exc:  # noqa: BLE001 - batch runs should report failures and keep moving.
                        totals["hours_skipped"] += 1
                        print(f"[hour-failed] {hour_key}: {exc}", flush=True)

        print(f"[day-done] {day.isoformat()} seconds={time.perf_counter() - day_start:.1f}", flush=True)
        if args.max_hours and totals["hours_done"] >= args.max_hours:
            print(f"[max-hours] reached {args.max_hours}; stopping", flush=True)
            break

    print(
        "[done] "
        f"days={totals['days']} hours_seen={totals['hours_seen']} "
        f"hours_done={totals['hours_done']} hours_skipped={totals['hours_skipped']} "
        f"files_downloaded={totals['files']} seconds={time.perf_counter() - run_start:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
