#!/usr/bin/env python3
"""
punch_build_background.py

The Universal Background Builder for PUNCH.
Consolidates spatial min/percentile binning, temporal percentile stacking, 
and optional morphological healing into a single script.

Outputs ONE text file representing the background over the requested time window.
"""

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
from scipy.ndimage import grey_closing
import argparse, glob, os, sys, warnings

warnings.filterwarnings("ignore")
S10_COEFF = 4.5e-16

# -------------------------
# FITS Helpers
# -------------------------
def _get_date_obs_any_hdu(hdul):
    d = hdul[0].header.get("DATE-OBS") if len(hdul) > 0 else None
    if (not d) and (len(hdul) > 1):
        d = hdul[1].header.get("DATE-OBS")
    return d

def get_timestamp_from_header(header):
    d = header.get("DATE-OBS")
    if d:
        return Time(d, format="isot", scale="utc").strftime("%Y%m%d%H%M%S")
    return "00000000000000"

def load_fits_data(fp):
    try:
        with fits.open(fp) as hdul:
            if hdul[0].data is not None:
                data, hdr = hdul[0].data, hdul[0].header
            elif len(hdul) > 1 and hdul[1].data is not None:
                data, hdr = hdul[1].data, hdul[1].header
            else:
                return None, None, None, None, None

            data = np.asarray(data).squeeze().astype(np.float64)
            date_obs = _get_date_obs_any_hdu(hdul) or hdr.get("DATE-OBS")
            if not date_obs: return None, None, None, None, None
            
            t = Time(date_obs, format="isot", scale="utc")
            wcs_solar = WCS(hdr)
            try: wcs_radec = WCS(hdr, key="A")
            except Exception: wcs_radec = None

            return data, t, wcs_solar, wcs_radec, hdr
    except Exception:
        return None, None, None, None, None

def parse_hhmm(s):
    hh, mm = s.split(":")
    return int(hh), int(mm)

def filter_files_by_time(files, t0, t1):
    out = []
    for fp in files:
        try:
            with fits.open(fp) as hdul:
                d = _get_date_obs_any_hdu(hdul)
                if not d: continue
                t = Time(d, format="isot", scale="utc")
                if (t >= t0) and (t < t1): out.append(fp)
        except Exception: continue
    return sorted(out)

# -------------------------
# Healing Logic
# -------------------------
def heal_dark_seams(image_data, seam_size=3):
    """Applies morphological closing to fill dark cracks."""
    temp_filled = image_data.copy()
    mask_nan = np.isnan(temp_filled)
    if np.all(mask_nan): return image_data
    
    global_median = np.nanmedian(temp_filled)
    temp_filled[mask_nan] = global_median
    healed = grey_closing(temp_filled, size=(seam_size, seam_size))
    healed[mask_nan] = np.nan
    return healed

# -------------------------
# Core Processing
# -------------------------
def process(files, args):
    print(f"Initializing grid using {os.path.basename(files[0])}...")
    data0, t0, wcs_solar0, wcs_radec0, hdr0 = load_fits_data(files[0])
    if data0 is None or wcs_solar0 is None or wcs_radec0 is None:
        print("Init failed (data/WCS missing).")
        return

    H, W = data0.shape
    y_idx, x_idx = np.indices((H, W))

    hpln0, hplt0 = wcs_solar0.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())
    x_bins = np.arange(np.floor(np.nanmin(hpln0)), np.ceil(np.nanmax(hpln0)) + args.bin_size, args.bin_size)
    y_bins = np.arange(np.floor(np.nanmin(hplt0)), np.ceil(np.nanmax(hplt0)) + args.bin_size, args.bin_size)

    nbx, nby = len(x_bins) - 1, len(y_bins) - 1
    bin_hpln = binned_statistic_2d(hpln0, hplt0, hpln0, statistic="mean", bins=[x_bins, y_bins]).statistic.T
    bin_hplt = binned_statistic_2d(hpln0, hplt0, hplt0, statistic="mean", bins=[x_bins, y_bins]).statistic.T

    start_ts = get_timestamp_from_header(hdr0)
    per_frame, per_time = [], []

    # Decide spatial statistic function
    if args.spatial_stat.lower() == 'min':
        stat_func = 'min'
    else:
        try:
            p_val = float(args.spatial_stat)
            stat_func = lambda v: np.percentile(v, p_val)
        except ValueError:
            print("Error: --spatial_stat must be 'min' or a number (e.g., '30.0').")
            sys.exit(1)

    print(f"Processing {len(files)} files...")
    for i, fp in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {os.path.basename(fp)}")
        data, tt, wcs_solar, _, hdr = load_fits_data(fp)
        if data is None or wcs_solar is None: continue
        
        last_hdr = hdr
        flat = (data.flatten() / S10_COEFF).astype(np.float64)
        hpln, hplt = wcs_solar.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())

        xi = np.digitize(hpln, x_bins) - 1
        yi = np.digitize(hplt, y_bins) - 1

        ok = (xi >= 0) & (xi < nbx) & (yi >= 0) & (yi < nby) & np.isfinite(flat) & (flat > 0) & (flat < 2000)

        grid = binned_statistic_2d(
            hpln[ok], hplt[ok], flat[ok],
            statistic=stat_func, bins=[x_bins, y_bins]
        ).statistic.T

        grid[np.isnan(grid)] = np.nan
        per_frame.append(grid)
        per_time.append(tt)

    if not per_frame:
        print("No usable frames processed.")
        return

    # Temporal Stacking
    print(f"Taking temporal {args.temporal_p}th percentile across {len(per_frame)} frames...")
    stack = np.stack(per_frame, axis=0)
    final_bg = np.nanpercentile(stack, args.temporal_p, axis=0)

    # Optional Healing
    if args.heal:
        print("Applying spatial seam healing...")
        final_bg = heal_dark_seams(final_bg, seam_size=3)

    # Time tracking
    diffs = np.abs(stack - final_bg[None, :, :])
    diffs[np.isnan(diffs)] = np.inf
    best_idx = np.argmin(diffs, axis=0)

    time_grid = np.full(final_bg.shape, "", dtype="<U30")
    for fi, tt in enumerate(per_time):
        time_grid[best_idx == fi] = tt.to_datetime().isoformat()

    # Output formatting
    print("Converting to RA/DEC...")
    res_s10 = final_bg.flatten()
    res_time = time_grid.flatten()
    
    px, py = wcs_solar0.world_to_pixel_values(bin_hpln.flatten(), bin_hplt.flatten())
    ra, dec = wcs_radec0.pixel_to_world_values(px, py)

    m = np.isfinite(res_s10) & (res_time != "") & np.isfinite(ra) & np.isfinite(dec)
    
    clean_ra, clean_dec, clean_s10, clean_time = ra[m], dec[m], res_s10[m], res_time[m]

    # File Naming
    end_ts = get_timestamp_from_header(last_hdr)
    t_dt = t0.to_datetime()
    jan1_jd = Time(t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0), scale="utc").jd
    header_line = f"{t_dt.year} {t0.jd - jan1_jd + 1.0:.8f}"

    heal_str = "_healed" if args.heal else ""
    out_name = f"PUNCH_L3_CIM_RANGE_{start_ts}_{end_ts}_s{args.spatial_stat}_t{int(args.temporal_p)}{heal_str}_bin.txt"

    print(f"Writing {len(clean_s10)} points -> {out_name}")
    with open(out_name, "w") as f:
        f.write(header_line + "\n")
        for r, d, b, tm in zip(clean_ra, clean_dec, clean_s10, clean_time):
            f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {tm}\n")
    print("Done.")

def main():
    ap = argparse.ArgumentParser(description="Universal PUNCH Background Map Builder")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--pattern", default="*_v0i.fits")
    ap.add_argument("--date", required=True)
    ap.add_argument("--start", default="00:00")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--bin_size", type=float, default=1.0)
    
    # The magical unified arguments
    ap.add_argument("--spatial_stat", type=str, default="min", 
                    help="Spatial binning stat: 'min' or a percentile number like '30.0'")
    ap.add_argument("--temporal_p", type=float, default=30.0, 
                    help="Temporal percentile across frames (e.g., 10.0, 25.0, 30.0)")
    ap.add_argument("--heal", action="store_true", 
                    help="Add this flag to apply morphological closing to dark seams")
    
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files: sys.exit(f"No files found matching {args.pattern} in {args.dir}")

    hh, mm = parse_hhmm(args.start)
    t0 = Time(f"{args.date}T{hh:02d}:{mm:02d}:00", format="isot", scale="utc")
    t1 = t0 + TimeDelta(args.hours * 3600.0, format="sec")

    files = filter_files_by_time(files, t0, t1)
    if not files: sys.exit("No files in time window.")
    
    process(files, args)

if __name__ == "__main__":
    main()