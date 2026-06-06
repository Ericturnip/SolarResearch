#!/usr/bin/env python3
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
    RemoteFile,
    CachedGrid,
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


DEFAULT_BASE_URL = "https://umbra.nascom.nasa.gov/punch/3/PTM"
DEFAULT_LAYER = "Polar_pB"


def remote_timestamp(name: str) -> datetime:
    match = re.search(r"PUNCH_L3_PTM_(\d{14})_", name)
    if not match:
        raise ValueError(f"Cannot parse PTM timestamp from {name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def list_day_files(day: date, base_url: str, timeout: float, retries: int) -> list[RemoteFile]:
    url = f"{base_url.rstrip('/')}/{day.year:04d}/{day.month:02d}/{day.day:02d}/"
    html = fetch_text(url, timeout=timeout, retries=retries)
    if html is None:
        return []

    names = sorted(set(re.findall(r'href="(PUNCH_L3_PTM_\d{14}_v0k\.fits)"', html)))
    out = []
    for name in names:
        try:
            out.append(RemoteFile(name=name, timestamp=remote_timestamp(name), url=urljoin(url, name)))
        except ValueError:
            continue
    return out


def safe_layer_name(layer: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", layer).strip("_")


def process_hour(
    hour_key: str,
    paths: list[Path],
    output_root: Path,
    *,
    bin_size_deg: float,
    layer: str,
) -> Path | None:
    if not paths:
        print(f"[skip-hour] {hour_key}: no downloaded FITS", flush=True)
        return None

    adapter = get_adapter("PTM")
    bmaps: list[BinnedMap] = []
    grid_cache: dict[str, CachedGrid] = {}
    for path in paths:
        try:
            frame = adapter.load_layer(path, layer)
            key = wcs_cache_key(frame)
            if key not in grid_cache:
                grid_cache[key] = prepare_grid(frame, bin_size_deg)
                print(f"[grid-cache] cached PTM grid #{len(grid_cache)} shape={grid_cache[key].shape}", flush=True)
            bmaps.append(median_bin_fast(frame, grid_cache[key]))
        except Exception as exc:  # noqa: BLE001 - bad frame should not stop the run.
            print(f"[process-file-failed] {path.name}: {exc}", flush=True)

    if not bmaps:
        print(f"[skip-hour] {hour_key}: no processable FITS", flush=True)
        return None

    comp = p25_composite(bmaps)
    dt = bmaps[0].timestamp.to_datetime()
    layer_slug = safe_layer_name(comp.layer_name or layer)
    day_dir = output_root / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"
    out_path = day_dir / f"PUNCH_L3_PTM_{dt.strftime('%Y%m%d%H')}_{layer_slug}_p25_COMPOSITE.txt"
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
    download_workers: int,
    timeout: float,
    retries: int,
    bin_size_deg: float,
    layer: str,
    keep_fits: bool,
) -> tuple[str, int, int, Path | None, float]:
    first_dt = hour_files[0].timestamp
    fits_dir = cache_root / first_dt.strftime("%Y/%m/%d/%H")
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
        bin_size_deg=bin_size_deg,
        layer=layer,
    )
    if not keep_fits and fits_dir.exists():
        shutil.rmtree(fits_dir, ignore_errors=True)
    return hour_key, len(paths), 1 if written is not None else 0, written, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and process PTM Oct/Nov hourly p25 composites into daily TXT folders."
    )
    parser.add_argument("--start-date", default="2025-10-01", help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", default="2025-11-30", help="YYYY-MM-DD inclusive")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "ptm_hourly_p25"))
    parser.add_argument("--cache-root", default=str(ROOT / "outputs" / "_ptm_download_cache"))
    parser.add_argument("--download-workers", type=int, default=8, help="Parallel downloads per hour")
    parser.add_argument("--hour-workers", type=int, default=1, help="Number of hours to download/process at once")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--bin-size-deg", type=float, default=1.0)
    parser.add_argument("--keep-fits", action="store_true", help="Keep downloaded FITS files after each hour")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess hours whose TXT already exists")
    parser.add_argument("--min-files-per-hour", type=int, default=1)
    parser.add_argument(
        "--hour-filter",
        default="",
        help="Optional regex matched against YYYYMMDDHH hour keys, useful for small test runs",
    )
    parser.add_argument("--max-hours", type=int, default=0, help="Optional cap on processed hours")
    args = parser.parse_args()

    start = parse_ymd(args.start_date)
    end = parse_ymd(args.end_date)
    output_root = Path(args.output_root).expanduser()
    cache_root = Path(args.cache_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[start] PTM p25 hourly composites {start}..{end} layer={args.layer} "
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
        remote_files = list_day_files(day, args.base_url, timeout=args.timeout, retries=args.retries)
        if not remote_files:
            print(f"[day-skip] {day.isoformat()}: no PTM files found", flush=True)
            continue

        hour_jobs = []
        for hour_key, hour_files in group_by_hour(remote_files).items():
            if hour_filter is not None and hour_filter.search(hour_key) is None:
                continue
            if args.max_hours and totals["hours_done"] >= args.max_hours:
                break
            totals["hours_seen"] += 1
            first_dt = hour_files[0].timestamp
            out_path = (
                output_root
                / f"{first_dt.year:04d}"
                / f"{first_dt.month:02d}"
                / f"{first_dt.day:02d}"
                / f"PUNCH_L3_PTM_{first_dt.strftime('%Y%m%d%H')}_{safe_layer_name(args.layer)}_p25_COMPOSITE.txt"
            )
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
                        download_workers=args.download_workers,
                        timeout=args.timeout,
                        retries=args.retries,
                        bin_size_deg=args.bin_size_deg,
                        layer=args.layer,
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
                    except Exception as exc:  # noqa: BLE001 - keep overnight run moving.
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
