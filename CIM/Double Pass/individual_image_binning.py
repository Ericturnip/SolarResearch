#!/usr/bin/env python3
"""
PUNCH L3: Per-8min cleaning using an hourly base map (LOS-preserving)
HARD-MASK VERSION: "only subtract bad pixels" (i.e., exclude them) and do NOT fill/replace.

UPDATES (v2):
  - Uses Difference Check (I - H) instead of Ratio (I/H) to handle negative backgrounds.
  - Uses Fast Median Filter (scipy) instead of generic_filter loop.
  - Adds Binary Dilation to clean "halos" around bad pixels.
  - Tighter thresholds (Z=4.0) for aggressive cleaning.
"""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
from scipy.ndimage import median_filter, binary_dilation  # UPDATED IMPORTS
import sys, os, glob, warnings

warnings.filterwarnings("ignore")

# -----------------------
# USER PARAMETERS
# -----------------------
BIN_SIZE_DEG = 1.0

# Convert to S10
S10_COEFF = 4.5e-16

# Pixel-level filter in S10 BEFORE binning
S10_MIN = -200    # Lowered slightly to allow for negative background fluctuations
S10_MAX = 800     # Increased slightly to catch the peak of outliers before filtering
EXCLUDE_EXACT_ZERO = True

# Per-frame binning statistic (within one 8-min image)
PER_FRAME_BIN_STAT = "median"

# Hourly base map across frames in that hour
BASE_PERCENTILE = 25

# Masking thresholds
MIN_BIN_COUNT = 1      # bins with fewer contributing pixels are masked
LOCAL_WIN = 5          # INCREASED to 5x5 for better structural awareness
Z_THRESH = 4.0         # TIGHTENED from 6.0 to 4.0 (Aggressive)

# Global Difference Threshold (New)
# Since background ranges from -56 to +140, a difference of +200 safely cuts artifacts.
GLOBAL_DIFF_THRESH = 150.0 

# Dilation (New)
# If True, expands the bad mask by 1 bin in all directions to catch "halos"
DILATE_MASK = False

# Optional: persistent seam mask built across ALL provided frames
BUILD_SEAM_MASK = True
SEAM_BAD_FRAC = 0.6

# Output
OUT_DIR = os.environ.get("OUT_DIR", "out_cleaned_per_frame_hardmask")


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

    # Bin centers
    bin_hpln_centers = binned_statistic_2d(
        flat_hpln, flat_hplt, flat_hpln, statistic="mean", bins=[x_bins, y_bins]
    ).statistic.T
    bin_hplt_centers = binned_statistic_2d(
        flat_hpln, flat_hplt, flat_hplt, statistic="mean", bins=[x_bins, y_bins]
    ).statistic.T

    return {
        "x_bins": x_bins, "y_bins": y_bins,
        "x_idx": x_idx, "y_idx": y_idx,
        "wcs_solar_ref": wcs_solar_ref, "wcs_radec_ref": wcs_radec_ref,
        "t_ref": t_ref, "header_ref": header_ref,
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

    vx = hpln[keep_pix]; vy = hplt[keep_pix]; vv = flat_s10[keep_pix]

    # Guard: if the pixel filter removes everything, skip gracefully.
    # This prevents SciPy from crashing with "zero-size array to reduction operation minimum".
    if vx.size == 0 or vy.size == 0 or vv.size == 0:
        ny = len(y_bins) - 1
        nx = len(x_bins) - 1
        I = np.full((ny, nx), np.nan, dtype=np.float32)
        N = np.zeros((ny, nx), dtype=np.float32)
        return I, N, diag

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
        if data is None: continue
        I, N, _ = bin_one_frame(data, wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
        invalid = (~np.isfinite(I)) | (N < MIN_BIN_COUNT)
        if invalid_count is None:
            invalid_count = invalid.astype(np.int32)
        else:
            invalid_count += invalid.astype(np.int32)
        total += 1

    if total == 0:
        return None

    frac = invalid_count / float(total)
    seam_mask = frac > SEAM_BAD_FRAC
    print(f"[SEAM MASK] total_frames={total}, seam_bins={int(np.sum(seam_mask))} ({100*np.mean(seam_mask):.2f}%)")
    return seam_mask


# -----------------------
# UPDATED Mask building
# -----------------------
def build_bad_mask(I_k, H_t, N_k, seam_mask=None):
    """
    Revised V2 Masking:
      1. Basic Invalidity (NaNs, low coverage)
      2. Global Difference Check (I - H > Threshold) -> Handles negative backgrounds
      3. Fast Median/MAD Z-score
      4. Binary Dilation (Halo removal)
    """
    eps = 1e-6
    # Calculate Residual
    # We use (I - H) for difference check, and (I/H - 1) for MAD if needed, 
    # but MAD works better on the pure Difference if H is unstable near zero.
    # Let's stick to normalized residual for MAD but be careful.
    
    # Actually, for MAD, let's use the pure DIFFERENCE map (I - H)
    # This avoids "divide by zero" issues when H ~ 0.
    Diff = I_k - H_t

    # 1. Basic Invalidity
    M = ~np.isfinite(I_k) | ~np.isfinite(H_t)
    M |= (N_k < MIN_BIN_COUNT)
    if seam_mask is not None:
        M |= seam_mask

    # 2. Global Difference Check (The "Sanity" Check)
    # If a pixel is massively brighter than the background, kill it.
    M |= (Diff > GLOBAL_DIFF_THRESH)

    # 3. Fast Local Outliers (MAD on the Difference Map)
    # We smooth the Difference map to find the "local trend"
    # Handling NaNs in median_filter is tricky, so we fill them temporarily.
    Diff_filled = Diff.copy()
    Diff_filled[~np.isfinite(Diff_filled)] = 0.0 # Just for the filter calculation

    Diff_med = median_filter(Diff_filled, size=LOCAL_WIN)
    abs_dev = np.abs(Diff_filled - Diff_med)
    
    # MAD map
    mad_map = median_filter(abs_dev, size=LOCAL_WIN)
    
    # Avoid zero-division noise
    mad_map = np.where(mad_map < 1e-4, 1e-4, mad_map)
    
    # Z-score
    Z = abs_dev / (1.4826 * mad_map)

    # Apply Threshold (mask valid pixels only)
    # We only care if Z is high AND the pixel was finite to begin with
    M |= (Z > Z_THRESH) & np.isfinite(Diff)

    # 4. Dilation (The "Halo" Clean Sweep)
    if DILATE_MASK:
        # Expands the True regions (bad pixels) by 1 step
        M = binary_dilation(M, structure=np.ones((3,3)))

    return M, Diff, Z

# -----------------------
# Output
# -----------------------
def write_ascii_points(output_path, t_frame, wcs_solar_ref, wcs_radec_ref,
                       bin_hpln_centers, bin_hplt_centers, values_map, point_time_iso):
    res_s10 = values_map.ravel()
    res_hpln = bin_hpln_centers.ravel()
    res_hplt = bin_hplt_centers.ravel()

    target_pix_x, target_pix_y = wcs_solar_ref.world_to_pixel_values(res_hpln, res_hplt)
    if wcs_radec_ref is None:
        raise RuntimeError("RA/DEC WCS missing.")
    res_ra, res_dec = wcs_radec_ref.pixel_to_world_values(target_pix_x, target_pix_y)

    valid_mask = np.isfinite(res_s10) & np.isfinite(res_ra) & np.isfinite(res_dec)
    
    # Use the current frame's time for the header instead of the global grid time
    header_date_string = year_doy_fraction_string(t_frame)

    with open(output_path, "w") as f:
        f.write(f"{header_date_string}\n")
        for r, d, b in zip(res_ra[valid_mask], res_dec[valid_mask], res_s10[valid_mask]):
            f.write(f"L3  {r:6.2f} {d:6.2f}{b:8.2f} {point_time_iso}\n")

    return int(np.sum(valid_mask))


# -----------------------
# Main
# -----------------------
def process_all(fits_files):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INIT] Grid from {os.path.basename(fits_files[0])}")
    grid = build_global_grid(fits_files[0], bin_size_deg=BIN_SIZE_DEG)
    
    x_idx = grid["x_idx"]; y_idx = grid["y_idx"]
    x_bins = grid["x_bins"]; y_bins = grid["y_bins"]
    wcs_solar = grid["wcs_solar_ref"]

    seam_mask = None
    if BUILD_SEAM_MASK:
        seam_mask = build_seam_mask_over_all_files(fits_files, grid)

    hour_groups = group_files_by_hour(fits_files)
    total_points = 0

    for hour_key, hour_list in hour_groups.items():
        print(f"\n[HOUR] {hour_key}  n_frames={len(hour_list)}")

        per_frame_maps = []
        for p in hour_list:
            data, t, _, _, _ = load_fits_data(p)
            if data is None: continue
            I, _, _ = bin_one_frame(data, wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            per_frame_maps.append(I)

        if not per_frame_maps:
            continue

        stack = np.stack(per_frame_maps, axis=0)
        H_t = np.nanpercentile(stack, BASE_PERCENTILE, axis=0).astype(np.float32)

        # --- DIAGNOSTICS BLOCK (Check your Background!) ---
        print("[DIAG] Hourly Base Map Stats (H_t):")
        valid_H = H_t[np.isfinite(H_t)]
        if len(valid_H) > 0:
            p = np.percentile(valid_H, [0, 50, 99, 100])
            print(f"  Min: {p[0]:.2f} | Med: {p[1]:.2f} | 99%: {p[2]:.2f} | Max: {p[3]:.2f}")
        else:
            print("  [WARNING] H_t is all NaN!")
        # --------------------------------------------------

        for p in hour_list:
            data, t, wcs_solar, _, header = load_fits_data(p)
            if data is None: continue

            I_k, N_k, diag = bin_one_frame(data, wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            
            # CALL UPDATED MASK BUILDER
            M_k, Diff_k, Z_k = build_bad_mask(I_k, H_t, N_k, seam_mask=seam_mask)

            C_k = I_k.copy()
            C_k[M_k] = np.nan

            # Report
            n_bins_bad = int(np.sum(M_k & np.isfinite(I_k)))
            print(f"    [FRAME] {os.path.basename(p)} -> bad_bins_masked={n_bins_bad}")

            frame_ts = get_timestamp_from_header(header)
            
            # Apply the desired naming convention directly
            out_name = f"PUNCH_L3_CIM_{frame_ts}_CLEANED.txt"
            
            pts = write_ascii_points(
                os.path.join(OUT_DIR, out_name),
                t_frame=t, # Pass the current frame's time to generate the correct header
                wcs_solar_ref=wcs_solar,
                wcs_radec_ref=grid["wcs_radec_ref"],
                bin_hpln_centers=grid["bin_hpln_centers"],
                bin_hplt_centers=grid["bin_hplt_centers"],
                values_map=C_k,
                point_time_iso=t.to_datetime().isoformat()
            )
            total_points += pts

    print("\n[DONE]")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fits_files = sys.argv[1:]
    else:
        fits_files = sorted(glob.glob("*.fits"))
    
    if not fits_files:
        raise SystemExit("No .fits files found.")
    
    process_all(fits_files)
