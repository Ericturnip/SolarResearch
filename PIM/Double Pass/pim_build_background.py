#!/usr/bin/env python3
"""
punch_pim_binning_p30_1hr_medianfirst.py

Pipeline for PUNCH PIM (Polarized Image) MZP datacubes.

Per file:
1. Load PIM cube (3, Y, X)
2. Extract M, Z, P layers
3. Map pixels into 1 deg x 1 deg HPLN/HPLT bins
4. Compute per-bin median for M, Z, P separately
5. Compute Q, U, and pB from the filtered M/Z/P bin maps
6. Record diagnostics for negative Q and negative U
7. Convert pB to S10
8. Keep only finite S10 in [-500, 2000)

Across the hour:
9. Stack per-frame maps
10. Take temporal p30 at each bin
11. Record the closest real frame timestamp for each bin
12. Convert bin centers to RA/Dec and write ASCII output
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

# ==========================================
# USER PARAMETERS / CONSTANTS
# ==========================================
BIN_SIZE_DEG = 1.0
TEMPORAL_PERCENTILE_DEFAULT = 30.0
S10_COEFF = 4.5e-16

# Final pB/S10 acceptance range after pB calculation
S10_MIN = -500.0
S10_MAX = 2000.0

# ==========================================
# POLARIZATION COEFFICIENTS (MZP)
# ==========================================
COEFF_Q_Z  =  4.0 / 3.0
COEFF_Q_PM = -2.0 / 3.0
COEFF_U_P  =  2.0 / np.sqrt(3.0)
COEFF_U_M  = -2.0 / np.sqrt(3.0)


# -------------------------
# Helpers
# -------------------------
def calculate_qu_pb_from_maps(m_map, z_map, p_map):
    """
    Compute Q, U, and pB from filtered M, Z, P maps.
    Negative M/Z/P values are allowed.
    Non-finite inputs stay NaN.
    """
    q_map = np.full(m_map.shape, np.nan, dtype=np.float64)
    u_map = np.full(m_map.shape, np.nan, dtype=np.float64)
    pb_map = np.full(m_map.shape, np.nan, dtype=np.float64)

    finite_mask = (
        np.isfinite(m_map) &
        np.isfinite(z_map) &
        np.isfinite(p_map)
    )

    if not np.any(finite_mask):
        return q_map, u_map, pb_map

    q_vals = COEFF_Q_Z * z_map[finite_mask] + COEFF_Q_PM * (
        p_map[finite_mask] + m_map[finite_mask]
    )
    u_vals = COEFF_U_P * p_map[finite_mask] + COEFF_U_M * m_map[finite_mask]

    q_map[finite_mask] = q_vals
    u_map[finite_mask] = u_vals
    pb_map[finite_mask] = np.sqrt(q_vals**2 + u_vals**2)

    return q_map, u_map, pb_map


def summarize_negative_stats(arr):
    """
    For a finite-valued array, report:
      - number of finite bins
      - number of negative bins
      - percent negative
      - average of negative values
      - average absolute magnitude of negative values
      - minimum (most negative) value
    """
    finite = np.isfinite(arr)
    n_finite = int(np.sum(finite))

    if n_finite == 0:
        return {
            "n_finite": 0,
            "n_negative": 0,
            "pct_negative": np.nan,
            "avg_negative": np.nan,
            "avg_negative_abs": np.nan,
            "min_negative": np.nan,
        }

    neg = finite & (arr < 0)
    n_negative = int(np.sum(neg))

    if n_negative == 0:
        return {
            "n_finite": n_finite,
            "n_negative": 0,
            "pct_negative": 0.0,
            "avg_negative": np.nan,
            "avg_negative_abs": np.nan,
            "min_negative": np.nan,
        }

    neg_vals = arr[neg]
    return {
        "n_finite": n_finite,
        "n_negative": n_negative,
        "pct_negative": 100.0 * n_negative / n_finite,
        "avg_negative": float(np.mean(neg_vals)),
        "avg_negative_abs": float(np.mean(np.abs(neg_vals))),
        "min_negative": float(np.min(neg_vals)),
    }


def get_timestamp_from_header(header):
    date_obs = header.get("DATE-OBS")
    if date_obs:
        t = Time(date_obs, format="isot", scale="utc")
        return t.strftime("%Y%m%d%H%M%S")
    return "00000000000000"


def _get_date_obs_any_hdu(hdul):
    date_obs = None
    if len(hdul) > 0:
        date_obs = hdul[0].header.get("DATE-OBS")
    if (not date_obs) and (len(hdul) > 1):
        date_obs = hdul[1].header.get("DATE-OBS")
    return date_obs


def load_fits_data_pim(input_fits):
    """
    Load 3D PIM data and WCS. Returns:
      data, time object, solar WCS, RA/DEC WCS, header

    Uses celestial WCS to ignore STOKES axis.
    """
    try:
        with fits.open(input_fits) as hdul:
            data = None
            header = None

            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 3:
                    data = hdu.data
                    header = hdu.header
                    break

            if data is None:
                return None, None, None, None, None

            data = np.asarray(data, dtype=np.float64)

            if data.ndim != 3 or data.shape[0] != 3:
                print(f"[WARN] {os.path.basename(input_fits)} has unexpected shape {data.shape}, expected (3, Y, X).")
                return None, None, None, None, None

            date_obs = _get_date_obs_any_hdu(hdul) or header.get("DATE-OBS")
            if not date_obs:
                return None, None, None, None, None

            t = Time(date_obs, format="isot", scale="utc")

            wcs_solar = WCS(header).celestial
            try:
                wcs_radec = WCS(header, key="A").celestial
            except Exception:
                wcs_radec = None

            return data, t, wcs_solar, wcs_radec, header

    except Exception as e:
        print(f"[load_fits_data_pim] Error processing {input_fits}: {e}")
        return None, None, None, None, None


def parse_hhmm(s: str):
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("start time must be HH:MM")
    hh = int(parts[0])
    mm = int(parts[1])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError("invalid HH:MM")
    return hh, mm


def filter_files_by_time(files, t0: Time, t1: Time):
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


def year_doy_fraction_string(t_ref):
    t_dt = t_ref.to_datetime()
    jan1_dt = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1_dt, scale="utc").jd
    doy_fraction = t_ref.jd - jan1_jd + 1.0
    return f"{t_dt.year} {doy_fraction:.8f}"


# -------------------------
# Binning helpers
# -------------------------
def per_bin_median(vx, vy, vv, x_bins, y_bins):
    """
    Fast per-bin median without generic_filter.
    Returns array shaped (ny, nx).
    """
    nx = len(x_bins) - 1
    ny = len(y_bins) - 1

    ix = np.digitize(vx, x_bins) - 1
    iy = np.digitize(vy, y_bins) - 1

    good = (
        (ix >= 0) & (ix < nx) &
        (iy >= 0) & (iy < ny) &
        np.isfinite(vv)
    )

    ix = ix[good]
    iy = iy[good]
    vv = vv[good]

    out = np.full(ny * nx, np.nan, dtype=np.float64)
    if vv.size == 0:
        return out.reshape((ny, nx))

    bid = iy * nx + ix
    order = np.argsort(bid)
    bid_s = bid[order]
    vv_s = vv[order]

    start = 0
    while start < bid_s.size:
        end = start + 1
        while end < bid_s.size and bid_s[end] == bid_s[start]:
            end += 1
        out[bid_s[start]] = np.median(vv_s[start:end])
        start = end

    return out.reshape((ny, nx))


def build_global_grid(ref_fits, bin_size_deg=1.0):
    data_init, t_ref, wcs_solar_ref, wcs_radec_ref, header_ref = load_fits_data_pim(ref_fits)
    if data_init is None or wcs_solar_ref is None or wcs_radec_ref is None:
        raise RuntimeError("Failed to initialize global grid from reference FITS.")

    _, h, w = data_init.shape
    y_idx, x_idx = np.indices((h, w))

    flat_hpln, flat_hplt = wcs_solar_ref.pixel_to_world_values(x_idx.ravel(), y_idx.ravel())

    finite_xy = np.isfinite(flat_hpln) & np.isfinite(flat_hplt)
    flat_hpln = flat_hpln[finite_xy]
    flat_hplt = flat_hplt[finite_xy]

    min_x, max_x = np.nanmin(flat_hpln), np.nanmax(flat_hpln)
    min_y, max_y = np.nanmin(flat_hplt), np.nanmax(flat_hplt)

    x_bins = np.arange(np.floor(min_x), np.ceil(max_x) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(min_y), np.ceil(max_y) + bin_size_deg, bin_size_deg)

    bin_hpln_centers = binned_statistic_2d(
        flat_hpln, flat_hplt, flat_hpln, statistic="mean", bins=[x_bins, y_bins]
    ).statistic.T

    bin_hplt_centers = binned_statistic_2d(
        flat_hpln, flat_hplt, flat_hplt, statistic="mean", bins=[x_bins, y_bins]
    ).statistic.T

    return {
        "x_bins": x_bins,
        "y_bins": y_bins,
        "x_idx": x_idx,
        "y_idx": y_idx,
        "wcs_solar_ref": wcs_solar_ref,
        "wcs_radec_ref": wcs_radec_ref,
        "t_ref": t_ref,
        "header_ref": header_ref,
        "bin_hpln_centers": bin_hpln_centers,
        "bin_hplt_centers": bin_hplt_centers,
    }


def median_filter_layers_in_degree_bins(data_cube, wcs_solar, x_idx, y_idx, x_bins, y_bins):
    """
    For each layer (M, Z, P):
      - map all finite pixels into 1x1 degree bins
      - compute per-bin median
    """
    flat_hpln, flat_hplt = wcs_solar.pixel_to_world_values(x_idx.ravel(), y_idx.ravel())
    finite_xy = np.isfinite(flat_hpln) & np.isfinite(flat_hplt)

    layer_maps = []
    diag = {}
    layer_names = ["M", "Z", "P"]

    for li, lname in enumerate(layer_names):
        flat_vals = data_cube[li].ravel()

        keep = finite_xy & np.isfinite(flat_vals)
        vx = flat_hpln[keep]
        vy = flat_hplt[keep]
        vv = flat_vals[keep]

        med_map = per_bin_median(vx, vy, vv, x_bins, y_bins)
        layer_maps.append(med_map)

        diag[f"{lname}_n_input_pix"] = int(np.sum(keep))
        diag[f"{lname}_n_bins_finite"] = int(np.sum(np.isfinite(med_map)))

    return layer_maps[0], layer_maps[1], layer_maps[2], diag


def make_per_frame_pb_map(data_cube, wcs_solar, x_idx, y_idx, x_bins, y_bins):
    """
    Per frame:
      1. median filter M/Z/P in 1x1 degree bins
      2. compute Q, U, pB from filtered M/Z/P
      3. record negative-Q and negative-U diagnostics
      4. convert pB to S10
      5. keep finite S10 in [-500, 2000)
    """
    m_map, z_map, p_map, diag = median_filter_layers_in_degree_bins(
        data_cube, wcs_solar, x_idx, y_idx, x_bins, y_bins
    )

    q_map, u_map, pb_map = calculate_qu_pb_from_maps(m_map, z_map, p_map)

    q_stats = summarize_negative_stats(q_map)
    u_stats = summarize_negative_stats(u_map)

    s10_map = pb_map / S10_COEFF
    valid = np.isfinite(s10_map) & (s10_map >= S10_MIN) & (s10_map < S10_MAX)

    out_map = np.full(s10_map.shape, np.nan, dtype=np.float64)
    out_map[valid] = s10_map[valid]

    diag["Q_n_finite"] = q_stats["n_finite"]
    diag["Q_n_negative"] = q_stats["n_negative"]
    diag["Q_pct_negative"] = q_stats["pct_negative"]
    diag["Q_avg_negative"] = q_stats["avg_negative"]
    diag["Q_avg_negative_abs"] = q_stats["avg_negative_abs"]
    diag["Q_min_negative"] = q_stats["min_negative"]

    diag["U_n_finite"] = u_stats["n_finite"]
    diag["U_n_negative"] = u_stats["n_negative"]
    diag["U_pct_negative"] = u_stats["pct_negative"]
    diag["U_avg_negative"] = u_stats["avg_negative"]
    diag["U_avg_negative_abs"] = u_stats["avg_negative_abs"]
    diag["U_min_negative"] = u_stats["min_negative"]

    diag["pB_n_bins_finite_before_cut"] = int(np.sum(np.isfinite(s10_map)))
    diag["pB_n_bins_kept_after_cut"] = int(np.sum(valid))

    return out_map, diag


# -------------------------
# Output
# -------------------------
def write_ascii_points(output_path, t_ref, wcs_solar_ref, wcs_radec_ref,
                       bin_hpln_centers, bin_hplt_centers, values_map, time_map):
    res_s10 = values_map.ravel()
    res_hpln = bin_hpln_centers.ravel()
    res_hplt = bin_hplt_centers.ravel()
    res_time = time_map.ravel()

    target_pix_x, target_pix_y = wcs_solar_ref.world_to_pixel_values(res_hpln, res_hplt)
    res_ra, res_dec = wcs_radec_ref.pixel_to_world_values(target_pix_x, target_pix_y)

    valid_mask = (
        np.isfinite(res_s10) &
        np.isfinite(res_ra) &
        np.isfinite(res_dec) &
        (res_time != "")
    )

    header_date_string = year_doy_fraction_string(t_ref)

    with open(output_path, "w") as f:
        f.write(f"{header_date_string}\n")
        for r, d, b, tm in zip(res_ra[valid_mask], res_dec[valid_mask], res_s10[valid_mask], res_time[valid_mask]):
            f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {tm}\n")

    return int(np.sum(valid_mask))


# -------------------------
# Core hourly logic
# -------------------------
def process_punch_pim_p30_to_txt(input_fits_list, bin_size_deg=1.0, temporal_percentile=30.0,
                                 out_dir="."):
    if not input_fits_list:
        print("Error: Input FITS list is empty.")
        return

    os.makedirs(out_dir, exist_ok=True)

    print(f"[INIT] Grid from {os.path.basename(input_fits_list[0])}")
    grid = build_global_grid(input_fits_list[0], bin_size_deg=bin_size_deg)

    x_idx = grid["x_idx"]
    y_idx = grid["y_idx"]
    x_bins = grid["x_bins"]
    y_bins = grid["y_bins"]

    wcs_solar_ref = grid["wcs_solar_ref"]
    wcs_radec_ref = grid["wcs_radec_ref"]
    bin_hpln_centers = grid["bin_hpln_centers"]
    bin_hplt_centers = grid["bin_hplt_centers"]

    start_timestamp = None
    end_timestamp = None

    per_frame_maps = []
    per_frame_times = []

    print(f"[PROCESS] {len(input_fits_list)} PIM FITS files")

    for i, input_fits in enumerate(input_fits_list):
        print(f"  [{i+1}/{len(input_fits_list)}] {os.path.basename(input_fits)}")

        data, t, wcs_curr, _, header = load_fits_data_pim(input_fits)
        if data is None or wcs_curr is None or header is None:
            print("    skipped: could not load data/WCS/header")
            continue

        ts = get_timestamp_from_header(header)
        if start_timestamp is None:
            start_timestamp = ts
        end_timestamp = ts

        frame_map, diag = make_per_frame_pb_map(
            data, wcs_curr, x_idx, y_idx, x_bins, y_bins
        )

        print(
            f"    bins kept after pB/S10 cut: {diag['pB_n_bins_kept_after_cut']} "
            f"(M bins={diag['M_n_bins_finite']}, Z bins={diag['Z_n_bins_finite']}, P bins={diag['P_n_bins_finite']})"
        )

        print(
            f"    Q negative: {diag['Q_n_negative']}/{diag['Q_n_finite']} "
            f"({diag['Q_pct_negative']:.2f}%)"
        )
        if diag["Q_n_negative"] > 0:
            print(
                f"      Q avg negative = {diag['Q_avg_negative']:.6e}, "
                f"|avg| = {diag['Q_avg_negative_abs']:.6e}, "
                f"most negative = {diag['Q_min_negative']:.6e}"
            )

        print(
            f"    U negative: {diag['U_n_negative']}/{diag['U_n_finite']} "
            f"({diag['U_pct_negative']:.2f}%)"
        )
        if diag["U_n_negative"] > 0:
            print(
                f"      U avg negative = {diag['U_avg_negative']:.6e}, "
                f"|avg| = {diag['U_avg_negative_abs']:.6e}, "
                f"most negative = {diag['U_min_negative']:.6e}"
            )

        if not np.any(np.isfinite(frame_map)):
            print("    skipped: no valid bins after filtering")
            continue

        per_frame_maps.append(frame_map)
        per_frame_times.append(t)

    if not per_frame_maps:
        print("No usable frames processed. Exiting.")
        return

    stack = np.stack(per_frame_maps, axis=0)

    valid_counts = np.sum(np.isfinite(stack), axis=0)

    p_grid = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    good_bins = valid_counts > 0

    if np.any(good_bins):
        p_grid[good_bins] = np.nanpercentile(
            stack[:, good_bins],
            temporal_percentile,
            axis=0
        )

    diffs = np.abs(stack - p_grid[None, :, :])
    diffs[~np.isfinite(diffs)] = np.inf

    best_idx = np.full(p_grid.shape, -1, dtype=int)
    if np.any(good_bins):
        best_idx[good_bins] = np.argmin(diffs[:, good_bins], axis=0)

    time_grid = np.full(p_grid.shape, "", dtype="<U30")
    for fi, tt in enumerate(per_frame_times):
        m = (best_idx == fi)
        time_grid[m] = tt.to_datetime().isoformat()

    p_grid[~good_bins] = np.nan
    time_grid[~good_bins] = ""

    if start_timestamp is None:
        start_timestamp = "00000000000000"
    if end_timestamp is None:
        end_timestamp = start_timestamp

    out_name = f"PUNCH_L3_PIM_RANGE_{start_timestamp}_{end_timestamp}_p{int(temporal_percentile)}_bin.txt"
    out_path = os.path.join(out_dir, out_name)

    npts = write_ascii_points(
        output_path=out_path,
        t_ref=per_frame_times[0],
        wcs_solar_ref=wcs_solar_ref,
        wcs_radec_ref=wcs_radec_ref,
        bin_hpln_centers=bin_hpln_centers,
        bin_hplt_centers=bin_hplt_centers,
        values_map=p_grid,
        time_map=time_grid
    )

    print(f"[WRITE] {out_name}  points={npts}")
    print("[DONE]")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="PUNCH PIM hourly temporal p30 map with 1x1 degree median-first filtering")
    ap.add_argument("--dir", type=str, default=".", help="Directory containing FITS (default: .)")
    ap.add_argument("--pattern", type=str, default="*PIM*_v0i.fits",
                    help="Glob pattern (default: '*PIM*_v0i.fits')")
    ap.add_argument("--date", type=str, required=True, help="UTC date YYYY-MM-DD")
    ap.add_argument("--start", type=str, default="00:00", help="UTC start time HH:MM (default 00:00)")
    ap.add_argument("--hours", type=float, default=1.0, help="Window length in hours (default 1.0)")
    ap.add_argument("--bin_size_deg", type=float, default=BIN_SIZE_DEG, help="Bin size in degrees (default 1.0)")
    ap.add_argument("--temporal_percentile", type=float, default=TEMPORAL_PERCENTILE_DEFAULT,
                    help="Temporal percentile over frames (default 30.0)")
    ap.add_argument("--out_dir", type=str, default=".", help="Output directory (default: .)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        print(f"No PIM FITS files found (dir={args.dir}, pattern={args.pattern}).")
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

    process_punch_pim_p30_to_txt(
        files,
        bin_size_deg=args.bin_size_deg,
        temporal_percentile=args.temporal_percentile,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()