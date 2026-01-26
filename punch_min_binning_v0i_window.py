#!/usr/bin/env python3
"""
punch_min_binning_v0i_window.py

Like your original min-binning script, but with two key behaviors:
1) Only consider the "i" version of files: *_v0i.fits (ignore *_v0h.fits)
2) Select files by DATE-OBS within a UTC time window (default: 00:00 for 6 hours)

Assumptions:
- Files are named like: PUNCH_L3_CIM_20251111000029_v0i.fits (or similar)
- DATE-OBS may live in either HDU0 or HDU1 header; we check both.
- WCS for RA/Dec is in key 'A' (same as your current pipeline).

Usage:

# From inside the directory that contains the FITS:
python punch_min_binning_v0i_window.py --dir . --date 2025-11-11 --start 00:00 --hours 6

# Or point to the downloaded directory:
python punch_min_binning_v0i_window.py --dir ./punch_2025-11-11 --date 2025-11-11 --start 00:00 --hours 6
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d

warnings.filterwarnings("ignore")


def _get_date_obs_any_hdu(hdul):
    """Return DATE-OBS from HDU0 or HDU1 (whichever has it), else None."""
    date_obs = None
    if len(hdul) > 0:
        date_obs = hdul[0].header.get("DATE-OBS")
    if (not date_obs) and (len(hdul) > 1):
        date_obs = hdul[1].header.get("DATE-OBS")
    return date_obs


def _get_data_and_header_any_hdu(hdul):
    """
    Prefer data from HDU0 if present, else HDU1.
    Return (data, header). If none, (None, None).
    """
    if len(hdul) > 0 and hdul[0].data is not None:
        return hdul[0].data, hdul[0].header
    if len(hdul) > 1 and hdul[1].data is not None:
        return hdul[1].data, hdul[1].header
    return None, None


def get_timestamp_from_header(header):
    """Extract YYYYMMDDHHMMSS from DATE-OBS in a header."""
    date_obs = header.get("DATE-OBS")
    if date_obs:
        t = Time(date_obs, format="isot", scale="utc")
        return t.strftime("%Y%m%d%H%M%S")
    return "00000000000000"


def load_fits_data(input_fits):
    """
    Load data and WCS. Returns data, time object, solar WCS, RA/DEC WCS, header.
    DATE-OBS may be in HDU0 or HDU1; we check both.
    """
    try:
        with fits.open(input_fits) as hdul:
            data, header = _get_data_and_header_any_hdu(hdul)
            if data is None or header is None:
                return None, None, None, None, None

            data = np.asarray(data).squeeze().astype(np.float64)

            date_obs = _get_date_obs_any_hdu(hdul)
            if not date_obs:
                # As a fallback, check the same header we used for WCS
                date_obs = header.get("DATE-OBS")
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


def select_files_in_time_window_v0i(fits_files, start_time: Time, end_time: Time):
    """
    Keep only *_v0i.fits files whose DATE-OBS is within [start_time, end_time).
    """
    selected = []
    for fp in fits_files:
        base = os.path.basename(fp)
        if ("_v0i" not in base) or (not base.endswith(".fits")):
            continue

        try:
            with fits.open(fp) as hdul:
                date_obs = _get_date_obs_any_hdu(hdul)
                if not date_obs:
                    continue
                t = Time(date_obs, format="isot", scale="utc")
                if (t >= start_time) and (t < end_time):
                    selected.append(fp)
        except Exception:
            continue

    return sorted(selected)


def process_punch_l3s_min_to_txt(input_fits_list, bin_size_deg=1.0):
    # --- CONSTANTS ---
    S10_COEFF = 4.5e-16

    if not input_fits_list:
        print("Error: Input FITS list is empty.")
        return

    # --- INITIALIZE ARRAYS (based on first usable image) ---
    first_ok = None
    for cand in input_fits_list:
        data_init, t_init, wcs_solar_init, wcs_radec_init, header_init = load_fits_data(cand)
        if data_init is not None and wcs_solar_init is not None and wcs_radec_init is not None:
            first_ok = (cand, data_init, t_init, wcs_solar_init, wcs_radec_init, header_init)
            break

    if first_ok is None:
        print("Could not initialize from any FITS file (missing data/WCS). Exiting.")
        return

    init_file, data_init, t_init, wcs_solar_init, wcs_radec_init, header_init = first_ok
    print(f"Initializing process using {os.path.basename(init_file)}...")

    height, width = data_init.shape

    # Generate pixel grids once
    y_idx, x_idx = np.indices((height, width))

    # For bins, we use solar coords (HPLN/HPLT) from init WCS
    flat_hpln, flat_hplt = wcs_solar_init.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())

    # Define bin edges
    min_x, max_x = np.min(flat_hpln), np.max(flat_hpln)
    min_y, max_y = np.min(flat_hplt), np.max(flat_hplt)

    x_bins = np.arange(np.floor(min_x), np.ceil(max_x) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(min_y), np.ceil(max_y) + bin_size_deg, bin_size_deg)

    num_x_bins = len(x_bins) - 1
    num_y_bins = len(y_bins) - 1

    # Master storage (rows=y, cols=x)
    min_s10_bin = np.full((num_y_bins, num_x_bins), np.inf, dtype=np.float64)
    min_time_bin = np.full((num_y_bins, num_x_bins), "", dtype="<U30")

    # Bin centers (transpose to (y,x))
    bin_hpln_centers = binned_statistic_2d(
        flat_hpln, flat_hplt, flat_hpln, statistic="mean", bins=[x_bins, y_bins]
    ).statistic.T
    bin_hplt_centers = binned_statistic_2d(
        flat_hpln, flat_hplt, flat_hplt, statistic="mean", bins=[x_bins, y_bins]
    ).statistic.T

    start_timestamp = get_timestamp_from_header(header_init)
    last_good_header = header_init

    print(f"Processing {len(input_fits_list)} FITS files...")

    for i, input_fits in enumerate(input_fits_list):
        print(f"  [{i+1}/{len(input_fits_list)}] Reading {os.path.basename(input_fits)}...")

        data, t, wcs_curr, _, header_curr = load_fits_data(input_fits)
        if data is None or wcs_curr is None or header_curr is None:
            continue

        last_good_header = header_curr

        flat_data = data.flatten()
        curr_hpln, curr_hplt = wcs_curr.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())
        flat_s10 = flat_data / S10_COEFF

        x_indices = np.digitize(curr_hpln, x_bins) - 1
        y_indices = np.digitize(curr_hplt, y_bins) - 1

        valid_pixel_mask = (
            (x_indices >= 0) & (x_indices < num_x_bins) &
            (y_indices >= 0) & (y_indices < num_y_bins) &
            (~np.isnan(flat_s10)) & (flat_s10 > 0) & (flat_s10 < 2000)
        )

        valid_s10_vals = flat_s10[valid_pixel_mask]

        current_img_min_grid = binned_statistic_2d(
            curr_hpln[valid_pixel_mask],
            curr_hplt[valid_pixel_mask],
            valid_s10_vals,
            statistic="min",
            bins=[x_bins, y_bins],
        ).statistic.T

        current_img_min_grid[np.isnan(current_img_min_grid)] = np.inf

        better_mask = current_img_min_grid < min_s10_bin
        min_s10_bin[better_mask] = current_img_min_grid[better_mask]
        min_time_bin[better_mask] = t.to_datetime().isoformat()

    end_timestamp = get_timestamp_from_header(last_good_header)

    print("Converting grid to RA/DEC...")

    res_s10 = min_s10_bin.flatten()
    res_hpln = bin_hpln_centers.flatten()
    res_hplt = bin_hplt_centers.flatten()
    res_time = min_time_bin.flatten()

    target_pix_x, target_pix_y = wcs_solar_init.world_to_pixel_values(res_hpln, res_hplt)
    res_ra, res_dec = wcs_radec_init.pixel_to_world_values(target_pix_x, target_pix_y)

    valid_mask = (
        (~np.isinf(res_s10)) &
        (~np.isnan(res_s10)) &
        (res_time != "") &
        (~np.isnan(res_ra)) &
        (~np.isnan(res_dec))
    )

    clean_ra = res_ra[valid_mask]
    clean_dec = res_dec[valid_mask]
    clean_s10 = res_s10[valid_mask]
    clean_time = res_time[valid_mask]

    t_dt = t_init.to_datetime()
    jan1_dt = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1_dt, scale="utc").jd
    doy_fraction = t_init.jd - jan1_jd + 1.0
    header_date_string = f"{t_dt.year} {doy_fraction:.8f}"

    output_filename = f"PUNCH_L3_CIM_RANGE_{start_timestamp}_{end_timestamp}_min_bin.txt"

    print(f"Writing {len(clean_s10)} points to {output_filename}...")

    with open(output_filename, "w") as f:
        f.write(f"{header_date_string}\n")
        for r, d, b, tm in zip(clean_ra, clean_dec, clean_s10, clean_time):
            f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {tm}\n")

    print("Done.")


def parse_hhmm(s: str):
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("start time must be HH:MM")
    hh = int(parts[0])
    mm = int(parts[1])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError("invalid HH:MM")
    return hh, mm


def main():
    parser = argparse.ArgumentParser(description="PUNCH L3 min-binning over a UTC time window (v0i only)")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing FITS (default: .)")
    parser.add_argument("--date", type=str, required=True, help="UTC date YYYY-MM-DD (e.g., 2025-11-11)")
    parser.add_argument("--start", type=str, default="00:00", help="UTC start time HH:MM (default 00:00)")
    parser.add_argument("--hours", type=float, default=6.0, help="Window length in hours (default 6.0)")
    parser.add_argument("--bin_size_deg", type=float, default=1.0, help="Bin size in degrees (default 1.0)")
    parser.add_argument(
        "--pattern",
        type=str,
        default="PUNCH_L3_CIM_*_v0i.fits",
        help="Glob pattern inside --dir (default: PUNCH_L3_CIM_*_v0i.fits)",
    )
    args = parser.parse_args()

    fits_files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not fits_files:
        print(f"No v0i FITS files found (dir={args.dir}, pattern={args.pattern}).")
        sys.exit(1)

    hh, mm = parse_hhmm(args.start)
    t0 = Time(f"{args.date}T{hh:02d}:{mm:02d}:00", format="isot", scale="utc")
    t1 = t0 + TimeDelta(args.hours * 3600.0, format="sec")

    print(f"Filtering files to UTC window: [{t0.isot}, {t1.isot})")
    fits_files = select_files_in_time_window_v0i(fits_files, t0, t1)

    print(f"Selected {len(fits_files)} v0i files in window.")
    if not fits_files:
        print("No files fall inside the requested time window.")
        sys.exit(1)

    process_punch_l3s_min_to_txt(fits_files, bin_size_deg=args.bin_size_deg)


if __name__ == "__main__":
    main()
