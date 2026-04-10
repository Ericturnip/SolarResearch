#!/usr/bin/env python3
"""
PUNCH L3 PIM: Direct pB Binning (With Diagnostics & Padding Filter)
Extracts M, Z, P layers, applies spatial median filtering to each separately, 
runs diagnostics on negative values, calculates Q, U, and pB, 
and outputs a per-frame text file with empty/NaN coordinates filtered out.
"""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
import sys, os, glob, warnings

warnings.filterwarnings("ignore")

# -----------------------
# USER PARAMETERS
# -----------------------
BIN_SIZE_DEG = 1.0
S10_COEFF = 4.5e-16

# Pixel-level filter in S10 BEFORE binning
S10_MIN = -1200    
S10_MAX = 1200     

# Polarization Coefficients
COEFF_Q_Z  =  4.0 / 3.0
COEFF_Q_PM = -2.0 / 3.0
COEFF_U_P  =  2.0 / np.sqrt(3.0)
COEFF_U_M  = -2.0 / np.sqrt(3.0)

OUT_DIR = os.environ.get("OUT_DIR", "out_binned_pB_direct")

# -----------------------
# Diagnostic Helper
# -----------------------
def print_negative_diagnostics(arr, name):
    """Calculates and prints the percentage and magnitude of negative values."""
    finite_mask = np.isfinite(arr)
    n_finite = np.sum(finite_mask)
    
    if n_finite == 0:
        print(f"      {name}: 0 valid bins")
        return

    neg_mask = finite_mask & (arr < 0)
    n_neg = np.sum(neg_mask)
    pct_neg = (n_neg / n_finite) * 100.0

    if n_neg == 0:
        print(f"      {name}: 0.00% negative (0 / {n_finite})")
    else:
        avg_neg = np.mean(arr[neg_mask])
        print(f"      {name}: {pct_neg:05.2f}% negative ({n_neg} / {n_finite}) | Avg Neg Value: {avg_neg:.2e}")


# -----------------------
# FITS / WCS helpers
# -----------------------
def load_pim_mzp_layers(input_fits):
    try:
        with fits.open(input_fits) as hdul:
            hdu = hdul[1] if len(hdul) > 1 and hdul[1].data is not None else hdul[0]
            if hdu.data.ndim != 3 or hdu.data.shape[0] != 3:
                return None, None, None, None, None, None, None

            m_layer = np.asarray(hdu.data[0], dtype=np.float64)
            z_layer = np.asarray(hdu.data[1], dtype=np.float64)
            p_layer = np.asarray(hdu.data[2], dtype=np.float64)
            
            header = hdu.header
            t = Time(header.get("DATE-OBS"), format="isot", scale="utc")
            wcs_solar = WCS(header).celestial 
            
            try:
                wcs_radec = WCS(header, key="A").celestial
            except:
                wcs_radec = None

            return m_layer, z_layer, p_layer, t, wcs_solar, wcs_radec, header
    except Exception as e:
        print(f"Error loading {input_fits}: {e}")
        return None, None, None, None, None, None, None

def build_global_grid(ref_fits, bin_size_deg=1.0):
    m, z, p, t_ref, wcs_solar_ref, wcs_radec_ref, header_ref = load_pim_mzp_layers(ref_fits)
    h, w = m.shape
    y_idx, x_idx = np.indices((h, w))

    flat_hpln, flat_hplt = wcs_solar_ref.pixel_to_world_values(x_idx.ravel(), y_idx.ravel())
    x_bins = np.arange(np.floor(np.nanmin(flat_hpln)), np.ceil(np.nanmax(flat_hpln)) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(np.nanmin(flat_hplt)), np.ceil(np.nanmax(flat_hplt)) + bin_size_deg, bin_size_deg)

    bin_hpln = binned_statistic_2d(flat_hpln, flat_hplt, flat_hpln, statistic="mean", bins=[x_bins, y_bins]).statistic.T
    bin_hplt = binned_statistic_2d(flat_hpln, flat_hplt, flat_hplt, statistic="mean", bins=[x_bins, y_bins]).statistic.T

    return {"x_bins": x_bins, "y_bins": y_bins, "x_idx": x_idx, "y_idx": y_idx,
            "wcs_solar": wcs_solar_ref, "wcs_radec": wcs_radec_ref,
            "hpln_centers": bin_hpln, "hplt_centers": bin_hplt}

# -----------------------
# Binning & Math helpers
# -----------------------
def per_bin_median(vx, vy, vv, x_bins, y_bins):
    nx, ny = len(x_bins) - 1, len(y_bins) - 1
    ix = np.digitize(vx, x_bins) - 1
    iy = np.digitize(vy, y_bins) - 1
    
    good = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(vv)
    ix, iy, vv = ix[good], iy[good], vv[good]
    
    out = np.full(ny * nx, np.nan, dtype=np.float64)
    if vv.size == 0:
        return out.reshape((ny, nx))
        
    bid = iy * nx + ix
    order = np.argsort(bid)
    bid_s, vv_s = bid[order], vv[order]

    start = 0
    while start < bid_s.size:
        end = start + 1
        while end < bid_s.size and bid_s[end] == bid_s[start]: end += 1
        out[bid_s[start]] = np.median(vv_s[start:end])
        start = end
    return out.reshape((ny, nx))

def calculate_qu_pb_from_maps(m_map, z_map, p_map):
    q_map = np.full(m_map.shape, np.nan, dtype=np.float64)
    u_map = np.full(m_map.shape, np.nan, dtype=np.float64)
    pb_map = np.full(m_map.shape, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(m_map) & np.isfinite(z_map) & np.isfinite(p_map)
    if not np.any(finite_mask):
        return q_map, u_map, pb_map

    q_vals = COEFF_Q_Z * z_map[finite_mask] + COEFF_Q_PM * (p_map[finite_mask] + m_map[finite_mask])
    u_vals = COEFF_U_P * p_map[finite_mask] + COEFF_U_M * m_map[finite_mask]

    q_map[finite_mask] = q_vals
    u_map[finite_mask] = u_vals
    pb_map[finite_mask] = np.sqrt(q_vals**2 + u_vals**2)
    
    return q_map, u_map, pb_map

def process_files(fits_files):
    os.makedirs(OUT_DIR, exist_ok=True)
    grid = build_global_grid(fits_files[0], BIN_SIZE_DEG)
    
    for p in fits_files:
        m, z, p_layer, t, wcs_solar, _, header = load_pim_mzp_layers(p)
        if m is None: continue

        print(f"Processing: {os.path.basename(p)}")
        
        hpln, hplt = wcs_solar.pixel_to_world_values(grid["x_idx"].ravel(), grid["y_idx"].ravel())
        vx, vy = hpln, hplt
        
        # 1. Apply spatial median filter
        m_binned = per_bin_median(vx, vy, m.ravel(), grid["x_bins"], grid["y_bins"])
        z_binned = per_bin_median(vx, vy, z.ravel(), grid["x_bins"], grid["y_bins"])
        p_binned = per_bin_median(vx, vy, p_layer.ravel(), grid["x_bins"], grid["y_bins"])

        # 2. Calculate Q, U, and pB
        q_binned, u_binned, binned_pB = calculate_qu_pb_from_maps(m_binned, z_binned, p_binned)
        
        # 3. Print Diagnostics
        print("   --- Diagnostics (Pre-S10 Filter) ---")
        print_negative_diagnostics(m_binned, "M Layer")
        print_negative_diagnostics(z_binned, "Z Layer")
        print_negative_diagnostics(p_binned, "P Layer")
        print_negative_diagnostics(q_binned, "Stokes Q")
        print_negative_diagnostics(u_binned, "Stokes U")
        print("   ------------------------------------")
        
        # 4. Convert to S10 and filter final outliers
        s10_map = binned_pB / S10_COEFF
        
        # Flatten arrays for 1D output
        res_s10 = s10_map.ravel()
        res_ra, res_dec = grid["wcs_radec"].pixel_to_world_values(
            *grid["wcs_solar"].world_to_pixel_values(grid["hpln_centers"].ravel(), grid["hplt_centers"].ravel())
        )
        
        # THE FIX: Explicitly require RA/Dec to be finite AND drop exact 0.0 padding
        valid_mask = (
            np.isfinite(res_s10) & 
            (res_s10 < S10_MAX) & 
            (res_s10 > S10_MIN) &
            (res_s10 != 0.0) &         # <--- Drops the padded 0s
            np.isfinite(res_ra) & 
            np.isfinite(res_dec)
        )
        
        ts = t.strftime("%Y%m%d%H%M%S")
        out_path = os.path.join(OUT_DIR, f"PUNCH_L3_PIM_pB_{ts}_BINNED.txt")
        
        with open(out_path, "w") as f:
            f.write(f"{t.to_datetime().year} {t.jd - Time(f'{t.to_datetime().year}-01-01').jd + 1.0:.8f}\n")
            for r, d, b in zip(res_ra[valid_mask], res_dec[valid_mask], res_s10[valid_mask]):
                f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {t.to_datetime().isoformat()}\n")

if __name__ == "__main__":
    files = sorted(glob.glob("*.fits")) if len(sys.argv) == 1 else sys.argv[1:]
    process_files(files)