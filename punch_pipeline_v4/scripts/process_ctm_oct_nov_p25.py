#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen, urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from punch_pipeline_v4.adapters.registry import get_adapter
from punch_pipeline_v4.models import BinnedMap
from punch_pipeline_v4.processing.binning import build_grid_from_wcs
from punch_pipeline_v4.processing.units import native_to_s10
from punch_pipeline_v4.writers.ascii import write_tomography_txt


DEFAULT_BASE_URL = "https://umbra.nascom.nasa.gov/punch/3/CTM"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    timestamp: datetime
    url: str


@dataclass
class CachedGrid:
    x_bins: np.ndarray
    y_bins: np.ndarray
    hpln_centers: np.ndarray
    hplt_centers: np.ndarray
    keep_flat: np.ndarray
    bin_id_ordered: np.ndarray
    value_indices_ordered: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    shape: tuple[int, int]


def parse_ymd(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def remote_timestamp(name: str) -> datetime:
    match = re.search(r"PUNCH_L3_CTM_(\d{14})_", name)
    if not match:
        raise ValueError(f"Cannot parse CTM timestamp from {name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def list_day_files(day: date, base_url: str, timeout: float, retries: int) -> list[RemoteFile]:
    url = f"{base_url.rstrip('/')}/{day.year:04d}/{day.month:02d}/{day.day:02d}/"
    html = fetch_text(url, timeout=timeout, retries=retries)
    if html is None:
        return []

    names = sorted(set(re.findall(r'href="(PUNCH_L3_CTM_\d{14}_v0k\.fits)"', html)))
    out = []
    for name in names:
        try:
            out.append(RemoteFile(name=name, timestamp=remote_timestamp(name), url=urljoin(url, name)))
        except ValueError:
            continue
    return out


def fetch_text(url: str, *, timeout: float, retries: int) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 404:
                print(f"[missing-day] {url}", flush=True)
                return None
            print(f"[list-error] attempt={attempt}/{retries} {url}: {exc}", flush=True)
        except (URLError, TimeoutError) as exc:
            print(f"[list-error] attempt={attempt}/{retries} {url}: {exc}", flush=True)
        time.sleep(min(2**attempt, 30))
    return None


def group_by_hour(files: list[RemoteFile]) -> dict[str, list[RemoteFile]]:
    hours: dict[str, list[RemoteFile]] = {}
    for item in files:
        key = item.timestamp.strftime("%Y%m%d%H")
        hours.setdefault(key, []).append(item)
    return {key: sorted(value, key=lambda f: f.timestamp) for key, value in sorted(hours.items())}


def download_one(remote: RemoteFile, path: Path, *, timeout: float, retries: int) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    for attempt in range(1, retries + 1):
        try:
            urlretrieve(remote.url, tmp)
            if tmp.stat().st_size == 0:
                raise RuntimeError("downloaded file has zero bytes")
            tmp.replace(path)
            return path
        except Exception as exc:  # noqa: BLE001 - overnight runner should keep going.
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise RuntimeError(f"failed downloading {remote.url}: {exc}") from exc
            print(f"[download-retry] attempt={attempt}/{retries} {remote.name}: {exc}", flush=True)
            time.sleep(min(2**attempt, 30))
    return path


def download_hour(
    files: list[RemoteFile],
    fits_dir: Path,
    *,
    workers: int,
    timeout: float,
    retries: int,
) -> list[Path]:
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(download_one, remote, fits_dir / remote.name, timeout=timeout, retries=retries): remote
            for remote in files
        }
        for future in as_completed(futures):
            remote = futures[future]
            try:
                paths.append(future.result())
            except Exception as exc:  # noqa: BLE001 - skip bad files, keep the night moving.
                print(f"[download-failed] {remote.name}: {exc}", flush=True)
    return sorted(paths)


def wcs_cache_key(frame) -> str:
    header = frame.solar_wcs.to_header_string()
    return hashlib.sha1(f"{frame.data.shape}|{header}".encode("utf-8")).hexdigest()


def prepare_grid(frame, bin_size_deg: float) -> CachedGrid:
    grid = build_grid_from_wcs(frame.solar_wcs, frame.data.shape, bin_size_deg=bin_size_deg)
    hpln, hplt = frame.solar_wcs.pixel_to_world_values(grid["x_idx"].ravel(), grid["y_idx"].ravel())
    nx = len(grid["x_bins"]) - 1
    ny = len(grid["y_bins"]) - 1
    ix = np.digitize(hpln, grid["x_bins"]) - 1
    iy = np.digitize(hplt, grid["y_bins"]) - 1
    finite_xy = np.isfinite(hpln) & np.isfinite(hplt)
    keep_flat = finite_xy & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    value_indices = np.flatnonzero(keep_flat)
    bin_ids = iy[keep_flat] * nx + ix[keep_flat]
    order = np.argsort(bin_ids)
    bin_id_ordered = bin_ids[order]
    value_indices_ordered = value_indices[order]
    starts = np.r_[0, np.flatnonzero(np.diff(bin_id_ordered)) + 1]
    ends = np.r_[starts[1:], bin_id_ordered.size]
    return CachedGrid(
        x_bins=grid["x_bins"],
        y_bins=grid["y_bins"],
        hpln_centers=grid["hpln_centers"],
        hplt_centers=grid["hplt_centers"],
        keep_flat=keep_flat,
        bin_id_ordered=bin_id_ordered,
        value_indices_ordered=value_indices_ordered,
        starts=starts,
        ends=ends,
        shape=(ny, nx),
    )


def median_bin_fast(frame, grid: CachedGrid) -> BinnedMap:
    vals = native_to_s10(frame.data).ravel()
    ordered_vals = vals[grid.value_indices_ordered]
    ordered_ids = grid.bin_id_ordered
    out = np.full(grid.shape[0] * grid.shape[1], np.nan, dtype=np.float64)

    for start, end in zip(grid.starts, grid.ends):
        chunk = ordered_vals[start:end]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            out[ordered_ids[start]] = np.median(chunk)

    values = out.reshape(grid.shape)
    time_map = np.full(values.shape, frame.timestamp.to_datetime().isoformat(), dtype="<U30")
    time_map[~np.isfinite(values)] = ""
    return BinnedMap(
        product=frame.product,
        layer_name=frame.layer_name,
        values=values,
        hpln_centers=grid.hpln_centers,
        hplt_centers=grid.hplt_centers,
        timestamp=frame.timestamp,
        solar_wcs=frame.solar_wcs,
        radec_wcs=frame.radec_wcs,
        time_map=time_map,
        unit="S10",
        metadata={"converted_to_s10_before_binning": True, "fast_cached_grid": True},
    )


def p25_composite(bmaps: list[BinnedMap]) -> BinnedMap:
    stack = np.stack([np.asarray(bmap.values, dtype=np.float64) for bmap in bmaps], axis=0)
    stack[np.isfinite(stack) & (stack == 0.0)] = np.nan
    with np.errstate(all="ignore"):
        values = np.nanpercentile(stack, 25.0, axis=0)

    # Preserve the timestamp nearest to each percentile value for the TXT Time column.
    diffs = np.abs(stack - values[None, :, :])
    diffs[~np.isfinite(diffs)] = np.inf
    good = np.any(np.isfinite(stack), axis=0) & np.isfinite(values)
    best_idx = np.full(values.shape, -1, dtype=int)
    if np.any(good):
        best_idx[good] = np.argmin(diffs[:, good], axis=0)
    time_map = np.full(values.shape, "", dtype="<U30")
    for i, bmap in enumerate(bmaps):
        source = bmap.time_map
        if source is None:
            source = np.full(values.shape, bmap.timestamp.to_datetime().isoformat(), dtype="<U30")
        mask = best_idx == i
        time_map[mask] = source[mask]

    first = bmaps[0]
    return BinnedMap(
        product=first.product,
        layer_name=first.layer_name,
        values=values,
        hpln_centers=first.hpln_centers,
        hplt_centers=first.hplt_centers,
        timestamp=first.timestamp,
        solar_wcs=first.solar_wcs,
        radec_wcs=first.radec_wcs,
        time_map=time_map,
        unit="S10",
        metadata={"composite_method": "p25", "drop_zero_before_stat": True},
    )


def process_hour(
    hour_key: str,
    paths: list[Path],
    output_root: Path,
    *,
    bin_size_deg: float,
) -> Path | None:
    if not paths:
        print(f"[skip-hour] {hour_key}: no downloaded FITS", flush=True)
        return None

    adapter = get_adapter("CTM")
    bmaps: list[BinnedMap] = []
    grid_cache: dict[str, CachedGrid] = {}
    for path in paths:
        try:
            frame = adapter.load_layer(path, "brightness")
            key = wcs_cache_key(frame)
            if key not in grid_cache:
                grid_cache[key] = prepare_grid(frame, bin_size_deg)
                print(f"[grid-cache] cached CTM grid #{len(grid_cache)} shape={grid_cache[key].shape}", flush=True)
            bmaps.append(median_bin_fast(frame, grid_cache[key]))
        except Exception as exc:  # noqa: BLE001 - bad frame should not stop the run.
            print(f"[process-file-failed] {path.name}: {exc}", flush=True)

    if not bmaps:
        print(f"[skip-hour] {hour_key}: no processable FITS", flush=True)
        return None

    comp = p25_composite(bmaps)
    dt = bmaps[0].timestamp.to_datetime()
    day_dir = output_root / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"
    out_path = day_dir / f"PUNCH_L3_CTM_{dt.strftime('%Y%m%d%H')}_brightness_p25_COMPOSITE.txt"
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
    )
    if not keep_fits and fits_dir.exists():
        shutil.rmtree(fits_dir, ignore_errors=True)
    return hour_key, len(paths), 1 if written is not None else 0, written, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and process CTM Oct/Nov hourly p25 composites into daily TXT folders."
    )
    parser.add_argument("--start-date", default="2025-10-01", help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", default="2025-11-30", help="YYYY-MM-DD inclusive")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "ctm_hourly_p25"))
    parser.add_argument("--cache-root", default=str(ROOT / "outputs" / "_ctm_download_cache"))
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
        f"[start] CTM p25 hourly composites {start}..{end} "
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
            print(f"[day-skip] {day.isoformat()}: no CTM files found", flush=True)
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
                / f"PUNCH_L3_CTM_{first_dt.strftime('%Y%m%d%H')}_brightness_p25_COMPOSITE.txt"
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
