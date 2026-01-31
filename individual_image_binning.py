#!/usr/bin/env python3
"""
PUNCH L3: Per-8min cleaning using an hourly base map (LOS-preserving)
HARD-MASK VERSION: "only subtract bad pixels" (i.e., exclude them) and do NOT fill/replace.

What this script does (chosen approach)
--------------------------------------
For each hour:
  1) Build an hourly base map H_t on the 1°×1° HPLN/HPLT grid using the frames in that hour.
     - Per-frame bin stat: MEDIAN per bin (robust; avoids single-pixel dropouts).
     - Hourly base: percentile across the per-frame binned maps (default p=25).

For each 8-minute frame k within the hour:
  2) Bin frame -> I_k (median per bin) + coverage N_k (count of contributing detector pixels per bin).
  3) Compare to base using residual: R_k = I_k / H_t - 1
  4) Build a "bad bin mask" M_k using:
        - invalid bins (NaN/inf, H_t==0)
        - low coverage bins (N_k < MIN_BIN_COUNT)
        - optional persistent seam mask (bins invalid in > SEAM_BAD_FRAC of all frames)
        - local residual outliers via MAD z-score
  5) Apply HARD MASK ONLY:
        C_k = I_k everywhere
        C_k[M_k] = NaN
     No inpainting, no fallback-to-base, no smoothing.
     This preserves your original footprint (e.g., trifold gaps remain gaps).

Outputs:
  - Per-frame ASCII files like your original format, but with masked bins dropped
    (NaNs are excluded by the final "valid_mask" when writing points).

Run:
  python punch_clean_per_frame_hardmask.py
  python punch_clean_per_frame_hardmask.py *.fits
"""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
from scipy.ndimage import generic_filter
import sys, os, glob, warnings

warnings.filterwarnings("ignore")

# -----------------------
# USER PARAMETERS
# -----------------------
BIN_SIZE_DEG = 1.0

# Convert to S10
S10_COEFF = 4.5e-16

# Pixel-level filter in S10 BEFORE binning
S10_MIN = -50
S10_MAX = 500
EXCLUDE_EXACT_ZERO = True

# Per-frame binning statistic (within one 8-min image)
PER_FRAME_BIN_STAT = "median"  # "median" (robust) or "mean" (faster but less robust)

# Hourly base map across frames in that hour
BASE_PERCENTILE = 25

# Masking thresholds
MIN_BIN_COUNT = 1      # bins with fewer contributing pixels are masked
LOCAL_WIN = 3          # neighborhood size for local median/MAD
Z_THRESH = 6.0         # MAD z-score threshold (increase => keep more bins)

# Optional: persistent seam mask built across ALL provided frames
BUILD_SEAM_MASK = True
SEAM_BAD_FRAC = 0.6    # bins invalid in >60% frames => seam

# Output
OUT_DIR = "out_cleaned_per_frame_hardmask"


# -----------------------
# FITS / WCS helpers
# -----------------------
def get_timestamp_from_header(header):
    date_obs = header.get("DATE-OBS")
    if date_obs:
        t = Time(date_obs, format="isot", scale="utc")
        return t.strftime("%Y%m%d%H%M%S")
    return "00000000000000"


def load_fits_data(input_fits):
    try:
        with fits.open(input_fits) as hdul:
            if hdul[0].data is not None:
                data = hdul[0].data
                header = hdul[0].header
            elif len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
                header = hdul[1].header
            else:
                return None, None, None, None, None

        data = np.asarray(data).squeeze().astype(np.float64)
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
        print(f"[load_fits_data] Error processing {input_fits}: {e}")
        return None, None, None, None, None


def year_doy_fraction_string(t_ref):
    t_dt = t_ref.to_datetime()
    jan1_dt = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1_dt).jd
    doy_fraction = t_ref.jd - jan1_jd + 1.0
    return f"{t_dt.year} {doy_fraction:.8f}"


def build_global_grid(ref_fits, bin_size_deg=1.0):
    data_init, t_ref, wcs_solar_ref, wcs_radec_ref, header_ref = load_fits_data(ref_fits)
    if data_init is None:
        raise RuntimeError("Failed to load reference FITS for grid init.")

    h, w = data_init.shape
    y_idx, x_idx = np.indices((h, w))

    flat_hpln, flat_hplt = wcs_solar_ref.pixel_to_world_values(x_idx.ravel(), y_idx.ravel())
    min_x, max_x = np.nanmin(flat_hpln), np.nanmax(flat_hpln)
    min_y, max_y = np.nanmin(flat_hplt), np.nanmax(flat_hplt)

    x_bins = np.arange(np.floor(min_x), np.ceil(max_x) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(min_y), np.ceil(max_y) + bin_size_deg, bin_size_deg)

    # Bin centers for output coords
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


# -----------------------
# Binning helpers
# -----------------------
def pixel_filter_s10(flat_s10):
    keep = ~np.isnan(flat_s10)
    keep &= (flat_s10 < S10_MAX) & (flat_s10 > S10_MIN)
    if EXCLUDE_EXACT_ZERO:
        keep &= (flat_s10 != 0)
    return keep


def per_bin_median(vx, vy, vv, x_bins, y_bins):
    """Median per bin via digitize+grouping."""
    nx = len(x_bins) - 1
    ny = len(y_bins) - 1

    ix = np.digitize(vx, x_bins) - 1
    iy = np.digitize(vy, y_bins) - 1
    good = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    ix = ix[good]; iy = iy[good]; vv = vv[good]
    out = np.full(ny * nx, np.nan, dtype=np.float32)
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
        out[bid_s[start]] = np.median(vv_s[start:end]).astype(np.float32)
        start = end

    return out.reshape((ny, nx))


def bin_one_frame(data, wcs_solar, x_idx, y_idx, x_bins, y_bins, stat="median"):
    flat_data = data.ravel()
    hpln, hplt = wcs_solar.pixel_to_world_values(x_idx.ravel(), y_idx.ravel())
    flat_s10 = flat_data / S10_COEFF

    keep_pix = pixel_filter_s10(flat_s10)
    diag = {
        "n_pix_total": int(flat_s10.size),
        "n_pix_kept": int(np.sum(keep_pix)),
        "n_pix_excluded": int(flat_s10.size - np.sum(keep_pix)),
    }

    vx = hpln[keep_pix]
    vy = hplt[keep_pix]
    vv = flat_s10[keep_pix]

    # Coverage N per bin
    N = binned_statistic_2d(vx, vy, vv, statistic="count", bins=[x_bins, y_bins]).statistic.T.astype(np.float32)

    if stat == "mean":
        I = binned_statistic_2d(vx, vy, vv, statistic="mean", bins=[x_bins, y_bins]).statistic.T.astype(np.float32)
    else:
        I = per_bin_median(vx, vy, vv, x_bins, y_bins).astype(np.float32)

    diag["n_bins_with_data"] = int(np.sum(np.isfinite(I)))
    return I, N, diag


# -----------------------
# Grouping + seam mask
# -----------------------
def group_files_by_hour(fits_list):
    groups = {}
    times = {}
    for p in fits_list:
        _, t, _, _, _ = load_fits_data(p)
        if t is None:
            continue
        dt = t.to_datetime()
        dt_hr = dt.replace(minute=0, second=0, microsecond=0)
        key = dt_hr.isoformat()
        groups.setdefault(key, []).append(p)
        times[p] = dt
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda fp: times.get(fp))
    return dict(sorted(groups.items()))


def build_seam_mask_over_all_files(fits_files, grid):
    x_bins = grid["x_bins"]; y_bins = grid["y_bins"]
    x_idx = grid["x_idx"]; y_idx = grid["y_idx"]

    invalid_count = None
    total = 0

    print("\n[SEAM MASK] Building persistent seam mask across ALL frames...")
    for p in fits_files:
        data, t, wcs_solar, _, _ = load_fits_data(p)
        if data is None:
            continue
        I, N, _ = bin_one_frame(data, wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
        invalid = (~np.isfinite(I)) | (N < MIN_BIN_COUNT)
        if invalid_count is None:
            invalid_count = invalid.astype(np.int32)
        else:
            invalid_count += invalid.astype(np.int32)
        total += 1

    if total == 0:
        print("[SEAM MASK] No valid frames; seam mask disabled.")
        return None

    frac = invalid_count / float(total)
    seam_mask = frac > SEAM_BAD_FRAC
    print(f"[SEAM MASK] total_frames={total}, seam_bins={int(np.sum(seam_mask))} ({100*np.mean(seam_mask):.2f}% of bins)")
    return seam_mask


# -----------------------
# Mask building (bad bins)
# -----------------------
def local_mad_filter(a, win=3):
    def mad_func(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        med = np.median(x)
        return np.median(np.abs(x - med))
    return generic_filter(a, mad_func, size=win, mode="nearest")


def build_bad_mask(I_k, H_t, N_k, seam_mask=None):
    """
    Returns:
      M (bool): bad bins to EXCLUDE (set to NaN)
      R (float): residual I/H - 1
      Z (float): local MAD z-score map
    """
    eps = 1e-6
    H_safe = H_t + eps
    R = (I_k / H_safe) - 1.0

    # invalid / unsafe
    M = ~np.isfinite(I_k) | ~np.isfinite(H_t) | (H_t == 0)

    # coverage
    M |= (N_k < MIN_BIN_COUNT)

    # seam
    if seam_mask is not None:
        M |= seam_mask

    # local outliers in residual
    R_med = generic_filter(R, np.nanmedian, size=LOCAL_WIN, mode="nearest")
    R_mad = local_mad_filter(R, win=LOCAL_WIN)
    R_mad = np.where((~np.isfinite(R_mad)) | (R_mad <= 0), np.nan, R_mad)

    Z = np.full_like(R, np.nan, dtype=np.float32)
    finite = np.isfinite(R) & np.isfinite(R_mad)
    Z[finite] = np.abs(R[finite] - R_med[finite]) / R_mad[finite]

    M |= (Z > Z_THRESH)

    return M, R, Z


# -----------------------
# Output
# -----------------------
def write_ascii_points(output_path, t_ref, wcs_solar_ref, wcs_radec_ref,
                       bin_hpln_centers, bin_hplt_centers, values_map, point_time_iso):
    res_s10 = values_map.ravel()
    res_hpln = bin_hpln_centers.ravel()
    res_hplt = bin_hplt_centers.ravel()

    # Convert bin centers to RA/DEC
    target_pix_x, target_pix_y = wcs_solar_ref.world_to_pixel_values(res_hpln, res_hplt)
    if wcs_radec_ref is None:
        raise RuntimeError("RA/DEC WCS (key='A') is missing in reference header.")
    res_ra, res_dec = wcs_radec_ref.pixel_to_world_values(target_pix_x, target_pix_y)

    valid_mask = np.isfinite(res_s10) & np.isfinite(res_ra) & np.isfinite(res_dec)
    header_date_string = year_doy_fraction_string(t_ref)

    with open(output_path, "w") as f:
        f.write(f"{header_date_string}\n")
        for r, d, b in zip(res_ra[valid_mask], res_dec[valid_mask], res_s10[valid_mask]):
            f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {point_time_iso}\n")

    return int(np.sum(valid_mask))


# -----------------------
# Main
# -----------------------
def process_all(fits_files):
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INIT] Building grid from {os.path.basename(fits_files[0])}")
    grid = build_global_grid(fits_files[0], bin_size_deg=BIN_SIZE_DEG)

    x_bins = grid["x_bins"]; y_bins = grid["y_bins"]
    x_idx = grid["x_idx"]; y_idx = grid["y_idx"]
    wcs_solar_ref = grid["wcs_solar_ref"]
    wcs_radec_ref = grid["wcs_radec_ref"]
    bin_hpln_centers = grid["bin_hpln_centers"]
    bin_hplt_centers = grid["bin_hplt_centers"]

    ny, nx = bin_hpln_centers.shape
    total_bins = nx * ny
    print(f"[INIT] Bin grid nx={nx}, ny={ny}, bin_size={BIN_SIZE_DEG} deg")

    seam_mask = None
    if BUILD_SEAM_MASK:
        seam_mask = build_seam_mask_over_all_files(fits_files, grid)

    hour_groups = group_files_by_hour(fits_files)
    print(f"\n[GROUP] Found {len(hour_groups)} hour blocks")

    total_frames = 0
    total_points = 0

    for hour_key, hour_list in hour_groups.items():
        print(f"\n[HOUR] {hour_key}  n_frames={len(hour_list)}")

        # Build base H_t from this hour's frames
        per_frame_maps = []
        hour_pix_total = 0
        hour_pix_kept = 0
        hour_bins_with_data = []

        for p in hour_list:
            data, t, wcs_solar, _, _ = load_fits_data(p)
            if data is None:
                continue
            I, N, diag = bin_one_frame(data, wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            per_frame_maps.append(I)
            hour_pix_total += diag["n_pix_total"]
            hour_pix_kept += diag["n_pix_kept"]
            hour_bins_with_data.append(diag["n_bins_with_data"])

        if len(per_frame_maps) == 0:
            print("  -> No valid frames in this hour; skip.")
            continue

        stack = np.stack(per_frame_maps, axis=0)
        H_t = np.nanpercentile(stack, BASE_PERCENTILE, axis=0).astype(np.float32)

        frac_pix_kept = hour_pix_kept / max(hour_pix_total, 1)
        print(f"  [HOUR DIAG] detector_pixels_kept={hour_pix_kept}/{hour_pix_total} ({100*frac_pix_kept:.2f}%)")
        print(f"  [HOUR DIAG] per-frame bins_with_data: min={np.min(hour_bins_with_data)}, "
              f"med={int(np.median(hour_bins_with_data))}, max={np.max(hour_bins_with_data)} / {total_bins}")
        print(f"  [HOUR DIAG] base finite bins: {int(np.sum(np.isfinite(H_t)))} / {total_bins}")

        # Clean each frame using HARD MASK ONLY
        for p in hour_list:
            data, t, wcs_solar, _, header = load_fits_data(p)
            if data is None:
                continue

            I_k, N_k, diag = bin_one_frame(data, wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            M_k, R_k, Z_k = build_bad_mask(I_k, H_t, N_k, seam_mask=seam_mask)

            # HARD MASK: do not fill, do not touch anything else
            C_k = I_k.copy()
            C_k[M_k] = np.nan

            # Diagnostics
            n_bins_data = int(np.sum(np.isfinite(I_k)))
            n_bins_bad = int(np.sum(M_k & np.isfinite(I_k)))
            frac_bad = n_bins_bad / max(n_bins_data, 1)

            bad_lowcov = int(np.sum((N_k < MIN_BIN_COUNT) & np.isfinite(I_k)))
            bad_outlier = int(np.sum((Z_k > Z_THRESH) & np.isfinite(Z_k)))
            bad_invalid = int(np.sum((~np.isfinite(I_k)) | (~np.isfinite(H_t)) | (H_t == 0)))

            print(f"    [FRAME] {os.path.basename(p)}")
            print(f"      pixels_kept={diag['n_pix_kept']}/{diag['n_pix_total']} ({100*diag['n_pix_kept']/max(diag['n_pix_total'],1):.2f}%) "
                  f"excluded={diag['n_pix_excluded']}")
            print(f"      bins_with_data={n_bins_data}/{total_bins}  bad_bins={n_bins_bad} ({100*frac_bad:.2f}%)")
            print(f"      bad_breakdown (counts may overlap): lowcov={bad_lowcov} outlier={bad_outlier} invalid={bad_invalid} "
                  f"{'(seam_enabled)' if seam_mask is not None else ''}")

            # Output (NaNs are dropped from ASCII)
            frame_ts = get_timestamp_from_header(header)
            hour_ts = hour_key.replace(":", "").replace("-", "").replace("T", "")[:10]  # YYYYMMDDHH
            out_name = f"PUNCH_L3_CIM_HARDMASK_{frame_ts}_HOUR_{hour_ts}_p{BASE_PERCENTILE}.txt"
            out_path = os.path.join(OUT_DIR, out_name)

            pts = write_ascii_points(
                out_path,
                t_ref=grid["t_ref"],
                wcs_solar_ref=wcs_solar_ref,
                wcs_radec_ref=wcs_radec_ref,
                bin_hpln_centers=bin_hpln_centers,
                bin_hplt_centers=bin_hplt_centers,
                values_map=C_k,
                point_time_iso=t.to_datetime().isoformat()
            )

            print(f"      wrote_points={pts} -> {out_name}")

            total_frames += 1
            total_points += pts

    print("\n[DONE]")
    print(f"  total_frames_processed={total_frames}")
    print(f"  total_points_written={total_points}")
    print(f"  output_dir={OUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fits_files = sys.argv[1:]
    else:
        fits_files = sorted(glob.glob("*.fits"))

    if not fits_files:
        raise SystemExit("No .fits files found / provided.")

    process_all(fits_files)
