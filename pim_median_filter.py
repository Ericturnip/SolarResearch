#!/usr/bin/env python3
"""
PUNCH L3 PIM: Direct pB Binning
Extracts the pre-calculated pB (Z-layer) from Polar_MZP FITS files
and applies spatial median filtering.
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
EXCLUDE_EXACT_ZERO = False

# Per-frame binning statistic
PER_FRAME_BIN_STAT = "median"

# Output
OUT_DIR = os.environ.get("OUT_DIR", "out_binned_pB_direct")

# -----------------------
# FITS / WCS helpers
# -----------------------
def load_pim_pB_layer(input_fits):
    """Loads only the pB layer (Index 1) from an MZP FITS file."""
    try:
        with fits.open(input_fits) as hdul:
            # PUNCH L3 PIM files often have the data in the first extension
            hdu = hdul[1] if len(hdul) > 1 else hdul[0]
            
            # Check for MZP mode
            obs_mode = hdu.header.get("OBS-MODE", "")
            if "MZP" not in obs_mode:
                print(f"[WARNING] {input_fits} is not in MZP mode. Expected pB at Index 1.")

            # Grab Index 1 (The 'Z' or polarized brightness layer)
            pB_data = np.asarray(hdu.data[1]).astype(np.float64)
            header = hdu.header
            
            t = Time(header.get("DATE-OBS"), format="isot", scale="utc")
            wcs_solar = WCS(header).celestial 
            
            try:
                wcs_radec = WCS(header, key="A").celestial
            except:
                wcs_radec = None

            return pB_data, t, wcs_solar, wcs_radec, header
    except Exception as e:
        print(f"Error loading {input_fits}: {e}")
        return None, None, None, None, None

def build_global_grid(ref_fits, bin_size_deg=1.0):
    data, t_ref, wcs_solar_ref, wcs_radec_ref, header_ref = load_pim_pB_layer(ref_fits)
    h, w = data.shape
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
# Binning helpers
# -----------------------
def per_bin_median(vx, vy, vv, x_bins, y_bins):
    nx, ny = len(x_bins) - 1, len(y_bins) - 1
    ix = np.digitize(vx, x_bins) - 1
    iy = np.digitize(vy, y_bins) - 1
    
    good = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, vv = ix[good], iy[good], vv[good]
    
    out = np.full(ny * nx, np.nan, dtype=np.float32)
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

def process_files(fits_files):
    os.makedirs(OUT_DIR, exist_ok=True)
    grid = build_global_grid(fits_files[0], BIN_SIZE_DEG)
    
    for p in fits_files:
        data, t, wcs_solar, _, header = load_pim_pB_layer(p)
        if data is None: continue

        print(f"Processing pB from {os.path.basename(p)}...")
        
        # Bin the pB layer directly
        hpln, hplt = wcs_solar.pixel_to_world_values(grid["x_idx"].ravel(), grid["y_idx"].ravel())
        flat_s10 = data.ravel() / S10_COEFF
        
        keep = ~np.isnan(flat_s10) & (flat_s10 < S10_MAX) & (flat_s10 > S10_MIN)
        if EXCLUDE_EXACT_ZERO: keep &= (flat_s10 != 0)
        
        binned_pB = per_bin_median(hpln[keep], hplt[keep], flat_s10[keep], grid["x_bins"], grid["y_bins"])

        # Write to ASCII
        ts = t.strftime("%Y%m%d%H%M%S")
        out_path = os.path.join(OUT_DIR, f"PUNCH_L3_PIM_pB_{ts}_BINNED.txt")
        
        res_s10 = binned_pB.ravel()
        res_ra, res_dec = grid["wcs_radec"].pixel_to_world_values(
            *grid["wcs_solar"].world_to_pixel_values(grid["hpln_centers"].ravel(), grid["hplt_centers"].ravel())
        )
        
        valid = np.isfinite(res_s10)
        with open(out_path, "w") as f:
            f.write(f"{t.to_datetime().year} {t.jd - Time(f'{t.to_datetime().year}-01-01').jd + 1.0:.8f}\n")
            for r, d, b in zip(res_ra[valid], res_dec[valid], res_s10[valid]):
                f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {t.to_datetime().isoformat()}\n")

if __name__ == "__main__":
    files = sorted(glob.glob("*.fits")) if len(sys.argv) == 1 else sys.argv[1:]
    process_files(files)