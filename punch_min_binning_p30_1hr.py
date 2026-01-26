#!/usr/bin/env python3
"""
punch_min_binning_p30_1hr.py

Like your original script, but instead of taking a strict min over time,
it takes a *temporal p30* (30th percentile over time) per 1° HPLN/HPLT bin
within a 1-hour window starting at a given UTC time.

Output format matches your min_bin.txt:
Line 1: "<year> <doy_fraction>"
Then:
L3  <RA> <Dec>  <S10> <ISO_TIME_OF_SELECTED_SAMPLE>

Notes:
- "p30 over 1 hour" is implemented by collecting one per-frame statistic per bin
  (here: per-frame bin MIN, like your original) and then taking the 30th percentile
  over frames for each bin.
- The timestamp we write is the frame time whose per-bin value is closest to the p30
  value (so you still get a representative time like your min-time output).
- DATE-OBS might be in HDU0 or HDU1; we check both.

Usage:
# Scan a directory for v0i fits, take 1-hour p30 starting 00:00 UTC
python punch_min_binning_p30_1hr.py --dir ./punch_2025-11-11 --date 2025-11-11 --start 00:00 --hours 1 --pattern "*_v0i.fits"

# From inside the dir:
python punch_min_binning_p30_1hr.py --dir . --date 2025-11-11 --start 00:00 --hours 1 --pattern "*_v0i.fits"
"""

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d

import argparse
import sys
import os
import glob
import warnings

warnings.filterwarnings("ignore")


# -------------------------
# Helpers
# -------------------------
def get_timestamp_from_header(header):
    """Extracts YYYYMMDDHHMMSS from DATE-OBS in a header."""
    date_obs = header.get("DATE-OBS")
    if date_obs:
        t = Time(date_obs, format="isot", scale="utc")
        return t.strftime("%Y%m%d%H%M%S")
    return "00000000000000"


def _get_date_obs_any_hdu(hdul):
    """Return DATE-OBS from HDU0 or HDU1 if present."""
    date_obs = None
    if len(hdul) > 0:
        date_obs = hdul[0].header.get("DATE-OBS")
    if (not date_obs) and (len(hdul) > 1):
        date_obs = hdul[1].header.get("DATE-OBS")
    return date_obs


def load_fits_data(input_fits):
    """
    Load data and WCS. Returns data, time object, solar WCS, RA/DEC WCS, header_used_for_wcs.
    """
    try:
        with fits.open(input_fits) as hdul:
            # Prefer HDU0 data; else HDU1
            if hdul[0].data is not None:
                data = hdul[0].data
                header = hdul[0].header
            elif len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
                header = hdul[1].header
            else:
                return None, None, None, None, None

            data = np.asarray(data).squeeze().astype(np.float64)

            date_obs = _get_date_obs_any_hdu(hdul) or header.get("DATE-OBS")
            if not date_obs:
                return None, None, None, None, None
            t = Time(date_obs, format="isot", scale="utc")

            wcs_solar = WCS(header)
            try:
                wcs_radec = WCS(header, key="A")
            except Exception:
                wcs_radec = None

            return data, t, wcs_solar, wcs_radec, header

    except Exception as e:
        print(f"Error processing {input_fits}: {e}")
        return None, None, None, None, None


def parse_hhmm(s: str):
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("start time must be HH:MM")
    hh = int(parts[0]); mm = int(parts[1])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError("invalid HH:MM")
    return hh, mm


def filter_files_by_time(files, t0: Time, t1: Time):
    """Keep files whose DATE-OBS is within [t0, t1)."""
    out = []
    for fp in files:
        try:
            with fits.open(fp) as hdul:
                date_obs = _get_date_obs_any_hdu(hdul)
                if not date_obs:
                    continue
                t = Time(date_obs, format="isot", scale="utc")
                if (t >= t0) and (t < t1):
                    out.append(fp)
        except Exception:
            continue
    return sorted(out)


# -------------------------
# Core logic
# -------------------------
def process_punch_l3s_p30_to_txt(input_fits_list, bin_size_deg=1.0, temporal_percentile=30.0):
    """
    For each FITS file, compute a per-bin statistic (current implementation uses per-frame MIN per bin,
    exactly like your min-map pipeline). Then compute the temporal p30 across frames for each bin.
    """
    S10_COEFF = 4.5e-16

    if not input_fits_list:
        print("Error: Input FITS list is empty.")
        return

    # Initialize using first usable file (need WCS + shape)
    print(f"Initializing process using {os.path.basename(input_fits_list[0])}...")
    data_init, t_init, wcs_solar_init, wcs_radec_init, header_init = load_fits_data(input_fits_list[0])
    if data_init is None or wcs_solar_init is None or wcs_radec_init is None:
        print("Could not initialize (missing data/WCS). Exiting.")
        return

    height, width = data_init.shape
    y_idx, x_idx = np.indices((height, width))

    # Bin edges from init frame solar coords
    flat_hpln0, flat_hplt0 = wcs_solar_init.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())

    min_x, max_x = np.nanmin(flat_hpln0), np.nanmax(flat_hpln0)
    min_y, max_y = np.nanmin(flat_hplt0), np.nanmax(flat_hplt0)

    x_bins = np.arange(np.floor(min_x), np.ceil(max_x) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(min_y), np.ceil(max_y) + bin_size_deg, bin_size_deg)

    num_x_bins = len(x_bins) - 1
    num_y_bins = len(y_bins) - 1

    # Bin centers (for output RA/DEC conversion)
    bin_hpln_centers = binned_statistic_2d(
        flat_hpln0, flat_hplt0, flat_hpln0, statistic="mean", bins=[x_bins, y_bins]
    ).statistic.T
    bin_hplt_centers = binned_statistic_2d(
        flat_hpln0, flat_hplt0, flat_hplt0, statistic="mean", bins=[x_bins, y_bins]
    ).statistic.T

    # Time tracking for filename
    start_timestamp = get_timestamp_from_header(header_init)
    end_timestamp = start_timestamp

    # We will store per-frame bin grids, then take temporal p30
    per_frame_grids = []
    per_frame_times = []

    print(f"Processing {len(input_fits_list)} FITS files...")

    last_good_header = header_init

    for i, input_fits in enumerate(input_fits_list):
        print(f"  [{i+1}/{len(input_fits_list)}] Reading {os.path.basename(input_fits)}...")

        data, t, wcs_curr, _, header_curr = load_fits_data(input_fits)
        if data is None or wcs_curr is None or header_curr is None:
            continue

        last_good_header = header_curr

        # Convert to S10
        flat_data = data.flatten()
        curr_hpln, curr_hplt = wcs_curr.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())
        flat_s10 = flat_data / S10_COEFF

        # Bin indices for validity mask
        x_indices = np.digitize(curr_hpln, x_bins) - 1
        y_indices = np.digitize(curr_hplt, y_bins) - 1

        valid_pixel_mask = (
            (x_indices >= 0) & (x_indices < num_x_bins) &
            (y_indices >= 0) & (y_indices < num_y_bins) &
            (~np.isnan(flat_s10)) & (flat_s10 > 0) & (flat_s10 < 2000)
        )

        valid_s10_vals = flat_s10[valid_pixel_mask]

        # Per-frame bin statistic (keep as MIN within bin to match your existing pipeline)
        frame_bin_grid = binned_statistic_2d(
            curr_hpln[valid_pixel_mask],
            curr_hplt[valid_pixel_mask],
            valid_s10_vals,
            statistic="min",
            bins=[x_bins, y_bins],
        ).statistic.T

        # Missing bins -> NaN
        frame_bin_grid[np.isnan(frame_bin_grid)] = np.nan

        per_frame_grids.append(frame_bin_grid)
        per_frame_times.append(t)

    end_timestamp = get_timestamp_from_header(last_good_header)

    if len(per_frame_grids) == 0:
        print("No usable frames processed. Exiting.")
        return

    # Stack: (nframes, nybins, nxbins)
    stack = np.stack(per_frame_grids, axis=0)

    # Temporal p30 for each bin
    p_grid = np.nanpercentile(stack, temporal_percentile, axis=0)

    # Choose a representative time per bin:
    # find frame whose value is closest to the p_grid (per bin)
    # We'll do this efficiently by computing abs diff over frames and argmin
    diffs = np.abs(stack - p_grid[None, :, :])
    diffs[np.isnan(diffs)] = np.inf
    best_idx = np.argmin(diffs, axis=0)  # shape (nybins, nxbins)

    # Build time grid as ISO strings
    time_grid = np.full(p_grid.shape, "", dtype="<U30")
    for fi, tt in enumerate(per_frame_times):
        m = best_idx == fi
        time_grid[m] = tt.to_datetime().isoformat()

    # --- Convert to RA/DEC ---
    print("Converting grid to RA/DEC...")

    res_s10 = p_grid.flatten()
    res_hpln = bin_hpln_centers.flatten()
    res_hplt = bin_hplt_centers.flatten()
    res_time = time_grid.flatten()

    target_pix_x, target_pix_y = wcs_solar_init.world_to_pixel_values(res_hpln, res_hplt)
    res_ra, res_dec = wcs_radec_init.pixel_to_world_values(target_pix_x, target_pix_y)

    valid_mask = (
        np.isfinite(res_s10) &
        (res_time != "") &
        np.isfinite(res_ra) &
        np.isfinite(res_dec)
    )

    clean_ra = res_ra[valid_mask]
    clean_dec = res_dec[valid_mask]
    clean_s10 = res_s10[valid_mask]
    clean_time = res_time[valid_mask]

    # --- Header date ---
    t_dt = t_init.to_datetime()
    jan1_dt = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1_dt, scale="utc").jd
    doy_fraction = t_init.jd - jan1_jd + 1.0
    header_date_string = f"{t_dt.year} {doy_fraction:.8f}"

    # --- Output filename ---
    output_filename = f"PUNCH_L3_CIM_RANGE_{start_timestamp}_{end_timestamp}_p{int(temporal_percentile)}_bin.txt"

    # --- Save ---
    print(f"Writing {len(clean_s10)} points to {output_filename}...")

    with open(output_filename, "w") as f:
        f.write(f"{header_date_string}\n")
        for r, d, b, tm in zip(clean_ra, clean_dec, clean_s10, clean_time):
            f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {tm}\n")

    print("Done.")


def main():
    ap = argparse.ArgumentParser(description="PUNCH L3 temporal p30 map over a time window (per-bin)")
    ap.add_argument("--dir", type=str, default=".", help="Directory containing FITS (default: .)")
    ap.add_argument("--pattern", type=str, default="*_v0i.fits",
                    help="Glob pattern (default: '*_v0i.fits' to avoid v0h)")
    ap.add_argument("--date", type=str, required=True, help="UTC date YYYY-MM-DD (e.g., 2025-11-11)")
    ap.add_argument("--start", type=str, default="00:00", help="UTC start time HH:MM (default 00:00)")
    ap.add_argument("--hours", type=float, default=1.0, help="Window length in hours (default 1.0)")
    ap.add_argument("--bin_size_deg", type=float, default=1.0, help="Bin size in degrees (default 1.0)")
    ap.add_argument("--temporal_percentile", type=float, default=30.0,
                    help="Temporal percentile over frames (default 30.0)")
    args = ap.parse_args()

    # Collect candidate files
    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        print(f"No FITS files found (dir={args.dir}, pattern={args.pattern}).")
        sys.exit(1)

    hh, mm = parse_hhmm(args.start)
    t0 = Time(f"{args.date}T{hh:02d}:{mm:02d}:00", format="isot", scale="utc")
    t1 = t0 + TimeDelta(args.hours * 3600.0, format="sec")

    print(f"Filtering files to UTC window: [{t0.isot}, {t1.isot})")
    files = filter_files_by_time(files, t0, t1)
    print(f"Selected {len(files)} files in window.")
    if not files:
        print("No files fall inside the requested time window.")
        sys.exit(1)

    process_punch_l3s_p30_to_txt(files, bin_size_deg=args.bin_size_deg, temporal_percentile=args.temporal_percentile)


if __name__ == "__main__":
    main()
