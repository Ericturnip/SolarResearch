#!/usr/bin/env python3
"""
PUNCH L3 PIM: Per-8min cleaning using an hourly base map
HARD-MASK VERSION with Pre-Combination Spatial Cleaning.

UPDATES for Stokes Parameters:
  - Loads 3 calibrated Stokes layers (I, Q, U) from the FITS datacube.
  - Bins (median filters) each layer independently to remove spatial outliers.
  - Combines cleaned states into Total Brightness (B = I) or Polarized Brightness (pB = sqrt(Q^2 + U^2)).
  - Applies the 1-hour temporal baseline mask to the combined product.
"""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
from scipy.ndimage import median_filter, binary_dilation
import sys, os, glob, warnings

warnings.filterwarnings("ignore")

# -----------------------
# USER PARAMETERS
# -----------------------
BIN_SIZE_DEG = 1.0
S10_COEFF = 4.5e-16

# What do you want to calculate from the 3 Stokes layers? ("pB" or "B")
TARGET_PRODUCT = "pB"

# Pixel-level filter in S10 BEFORE binning (applied to individual I, Q, U)
S10_MIN = -1200    
S10_MAX = 1200     
EXCLUDE_EXACT_ZERO = False

# Per-frame binning statistic
PER_FRAME_BIN_STAT = "median"

# Hourly base map across combined frames in that hour
BASE_PERCENTILE = 25

# Masking thresholds
MIN_BIN_COUNT = 1      
LOCAL_WIN = 5          
Z_THRESH = 5.0         
GLOBAL_DIFF_THRESH = 250.0 
DILATE_MASK = False

# Output
OUT_DIR = os.environ.get("OUT_DIR", "out_cleaned_pim")

# -----------------------
# FITS / WCS helpers
# -----------------------
def get_timestamp_from_header(header):
    date_obs = header.get("DATE-OBS")
    if date_obs:
        t = Time(date_obs, format="isot", scale="utc")
        return t.strftime("%Y%m%d%H%M%S")
    return "00000000000000"

def load_pim_fits_data(input_fits):
    """
    Loads PIM data, expecting either a (3, H, W) array or multiple extensions.
    Returns: data_cube (shape 3,H,W), time, wcs_solar, wcs_radec, header
    """
    try:
        with fits.open(input_fits) as hdul:
            data_cube = None
            header = None
            
            # Check if primary or first extension has the 3D cube (Stokes I, Q, U)
            for hdu in hdul:
                if hdu.data is not None:
                    if hdu.data.ndim == 3 and hdu.data.shape[0] == 3:
                        data_cube = np.asarray(hdu.data).astype(np.float64)
                        header = hdu.header
                        break
            
            # Fallback: if they are stored as 3 separate 2D extensions
            if data_cube is None and len(hdul) >= 4:
                d_I = np.asarray(hdul[1].data).astype(np.float64)
                d_Q = np.asarray(hdul[2].data).astype(np.float64)
                d_U = np.asarray(hdul[3].data).astype(np.float64)
                data_cube = np.stack([d_I, d_Q, d_U], axis=0)
                header = hdul[1].header

            if data_cube is None:
                print(f"[load_pim_fits_data] Could not find 3 Stokes layers in {input_fits}")
                return None, None, None, None, None

        date_obs = header.get("DATE-OBS")
        if not date_obs:
            return None, None, None, None, None

        t = Time(date_obs, format="isot", scale="utc")
        wcs_solar = WCS(header).celestial 

        try:
            wcs_radec = WCS(header, key="A").celestial
        except Exception:
            wcs_radec = None

        return data_cube, t, wcs_solar, wcs_radec, header
    except Exception as e:
        print(f"[load_pim_fits_data] Error processing {input_fits}: {e}")
        return None, None, None, None, None

def year_doy_fraction_string(t_ref):
    t_dt = t_ref.to_datetime()
    jan1_dt = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1_dt).jd
    doy_fraction = t_ref.jd - jan1_jd + 1.0
    return f"{t_dt.year} {doy_fraction:.8f}"

def build_global_grid(ref_fits, bin_size_deg=1.0):
    data_cube, t_ref, wcs_solar_ref, wcs_radec_ref, header_ref = load_pim_fits_data(ref_fits)
    if data_cube is None:
        raise RuntimeError("Failed to load reference FITS for grid init.")

    h, w = data_cube.shape[1], data_cube.shape[2]
    y_idx, x_idx = np.indices((h, w))

    flat_hpln, flat_hplt = wcs_solar_ref.pixel_to_world_values(x_idx.ravel(), y_idx.ravel())
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
        "x_bins": x_bins, "y_bins": y_bins,
        "x_idx": x_idx, "y_idx": y_idx,
        "wcs_solar_ref": wcs_solar_ref, "wcs_radec_ref": wcs_radec_ref,
        "t_ref": t_ref, "header_ref": header_ref,
        "bin_hpln_centers": bin_hpln_centers,
        "bin_hplt_centers": bin_hplt_centers,
    }

# -----------------------
# Binning & Math helpers
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

def bin_one_layer(layer_data, wcs_solar, x_idx, y_idx, x_bins, y_bins, stat="median"):
    flat_data = layer_data.ravel()
    hpln, hplt = wcs_solar.pixel_to_world_values(x_idx.ravel(), y_idx.ravel())
    flat_s10 = flat_data / S10_COEFF

    keep_pix = pixel_filter_s10(flat_s10)
    vx = hpln[keep_pix]; vy = hplt[keep_pix]; vv = flat_s10[keep_pix]

    ny = len(y_bins) - 1
    nx = len(x_bins) - 1
    if vx.size == 0 or vy.size == 0 or vv.size == 0:
        I = np.full((ny, nx), np.nan, dtype=np.float32)
        N = np.zeros((ny, nx), dtype=np.float32)
        return I, N

    N = binned_statistic_2d(vx, vy, vv, statistic="count", bins=[x_bins, y_bins]).statistic.T.astype(np.float32)

    if stat == "mean":
        I = binned_statistic_2d(vx, vy, vv, statistic="mean", bins=[x_bins, y_bins]).statistic.T.astype(np.float32)
    else:
        I = per_bin_median(vx, vy, vv, x_bins, y_bins).astype(np.float32)

    return I, N

def combine_polarization(I_stokes, Q_stokes, U_stokes, target="pB"):
    """Calculates Total Brightness (B) or Polarized Brightness (pB) directly from Stokes I, Q, U"""
    if target == "B":
        return I_stokes
    elif target == "pB":
        return np.sqrt(Q_stokes**2 + U_stokes**2)
    else:
        raise ValueError("TARGET_PRODUCT must be 'B' or 'pB'")

def group_files_by_hour(fits_list):
    groups = {}
    times = {}
    for p in fits_list:
        _, t, _, _, _ = load_pim_fits_data(p)
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

# -----------------------
# Mask building
# -----------------------
def build_bad_mask(Combined_k, H_t, N_min_layer):
    Diff = Combined_k - H_t

    # 1. Basic Invalidity (If any layer had low counts, mask the combined result)
    M = ~np.isfinite(Combined_k) | ~np.isfinite(H_t)
    M |= (N_min_layer < MIN_BIN_COUNT)

    # 2. Global Difference Check
    M |= (Diff > GLOBAL_DIFF_THRESH)

    # 3. Fast Local Outliers (MAD on the Difference Map)
    Diff_filled = Diff.copy()
    Diff_filled[~np.isfinite(Diff_filled)] = 0.0

    Diff_med = median_filter(Diff_filled, size=LOCAL_WIN)
    abs_dev = np.abs(Diff_filled - Diff_med)
    
    mad_map = median_filter(abs_dev, size=LOCAL_WIN)
    mad_map = np.where(mad_map < 1e-4, 1e-4, mad_map)
    Z = abs_dev / (1.4826 * mad_map)

    M |= (Z > Z_THRESH) & np.isfinite(Diff)

    if DILATE_MASK:
        M = binary_dilation(M, structure=np.ones((3,3)))

    return M

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

    hour_groups = group_files_by_hour(fits_files)
    total_points = 0

    for hour_key, hour_list in hour_groups.items():
        print(f"\n[HOUR] {hour_key}  n_frames={len(hour_list)}")

        per_frame_maps = []
        for p in hour_list:
            data_cube, t, _, _, _ = load_pim_fits_data(p)
            if data_cube is None: continue
            
            # Bin each Stokes layer separately
            I_stokes, N_I = bin_one_layer(data_cube[0], wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            Q_stokes, N_Q = bin_one_layer(data_cube[1], wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            U_stokes, N_U = bin_one_layer(data_cube[2], wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            
            Combined = combine_polarization(I_stokes, Q_stokes, U_stokes, target=TARGET_PRODUCT)
            per_frame_maps.append(Combined)

        if not per_frame_maps:
            continue

        stack = np.stack(per_frame_maps, axis=0)
        H_t = np.nanpercentile(stack, BASE_PERCENTILE, axis=0).astype(np.float32)

        print(f"[DIAG] Hourly Base Map Stats (H_t) for {TARGET_PRODUCT}:")
        valid_H = H_t[np.isfinite(H_t)]
        if len(valid_H) > 0:
            p_vals = np.percentile(valid_H, [0, 50, 99, 100])
            print(f"  Min: {p_vals[0]:.2f} | Med: {p_vals[1]:.2f} | 99%: {p_vals[2]:.2f} | Max: {p_vals[3]:.2f}")
        else:
            print("  [WARNING] H_t is all NaN!")

        for p in hour_list:
            data_cube, t, wcs_solar, _, header = load_pim_fits_data(p)
            if data_cube is None: continue

            # First pass: Bin Stokes parameters
            I_stokes, N_I = bin_one_layer(data_cube[0], wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            Q_stokes, N_Q = bin_one_layer(data_cube[1], wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            U_stokes, N_U = bin_one_layer(data_cube[2], wcs_solar, x_idx, y_idx, x_bins, y_bins, stat=PER_FRAME_BIN_STAT)
            
            # Find minimum valid pixel count across all 3 layers
            N_min_layer = np.minimum(np.minimum(N_I, N_Q), N_U)
            
            Combined_k = combine_polarization(I_stokes, Q_stokes, U_stokes, target=TARGET_PRODUCT)
            
            # Second pass: Mask against hourly baseline
            M_k = build_bad_mask(Combined_k, H_t, N_min_layer)

            C_k = Combined_k.copy()
            C_k[M_k] = np.nan

            n_bins_bad = int(np.sum(M_k & np.isfinite(Combined_k)))
            print(f"    [FRAME] {os.path.basename(p)} -> bad_bins_masked={n_bins_bad}")

            frame_ts = get_timestamp_from_header(header)
            out_name = f"PUNCH_L3_PIM_{TARGET_PRODUCT}_{frame_ts}_CLEANED.txt"
            
            pts = write_ascii_points(
                os.path.join(OUT_DIR, out_name),
                t_frame=t, 
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