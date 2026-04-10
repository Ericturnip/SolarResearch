#!/usr/bin/env python3
"""
PUNCH L3 CIM: Direct Spatial Median Binning
Reads 2D CIM FITS files, applies a fast 1x1 degree spatial median filter, 
and outputs a per-frame text file for tomography.
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

# Pixel-level filter in S10 BEFORE binning (allows normal noise, cuts extreme bad data)
S10_MIN = -50.0    
S10_MAX = 2000.0     

# Output directory
OUT_DIR = os.environ.get("OUT_DIR", "out_binned_cim_direct")

# -----------------------
# FITS / WCS helpers
# -----------------------
def load_cim_data(input_fits):
    """Loads 2D intensity layer from a CIM FITS file."""
    try:
        with fits.open(input_fits) as hdul:
            # CIM data can be in HDU 0 or HDU 1 depending on processing level
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            
            data = np.asarray(hdu.data).squeeze().astype(np.float64)
            header = hdu.header
            
            date_obs = header.get("DATE-OBS")
            if not date_obs:
                # Fallback if DATE-OBS is only in the primary header
                date_obs = hdul[0].header.get("DATE-OBS")
                
            t = Time(date_obs, format="isot", scale="utc")
            wcs_solar = WCS(header).celestial 
            
            try:
                wcs_radec = WCS(header, key="A").celestial
            except:
                wcs_radec = None

            return data, t, wcs_solar, wcs_radec, header
    except Exception as e:
        print(f"Error loading {input_fits}: {e}")
        return None, None, None, None, None

def build_global_grid(ref_fits, bin_size_deg=1.0):
    """Builds the 1x1 degree coordinate bins based on the first image."""
    data, t_ref, wcs_solar_ref, wcs_radec_ref, header_ref = load_cim_data(ref_fits)
    h, w = data.shape
    y_idx, x_idx = np.indices((h, w))

    flat_hpln, flat_hplt = wcs_solar_ref.pixel_to_world_values(x_idx.ravel(), y_idx.ravel())
    
    # Establish bin edges
    x_bins = np.arange(np.floor(np.nanmin(flat_hpln)), np.ceil(np.nanmax(flat_hpln)) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(np.nanmin(flat_hplt)), np.ceil(np.nanmax(flat_hplt)) + bin_size_deg, bin_size_deg)

    # Establish bin centers
    bin_hpln = binned_statistic_2d(flat_hpln, flat_hplt, flat_hpln, statistic="mean", bins=[x_bins, y_bins]).statistic.T
    bin_hplt = binned_statistic_2d(flat_hpln, flat_hplt, flat_hplt, statistic="mean", bins=[x_bins, y_bins]).statistic.T

    return {"x_bins": x_bins, "y_bins": y_bins, "x_idx": x_idx, "y_idx": y_idx,
            "wcs_solar": wcs_solar_ref, "wcs_radec": wcs_radec_ref,
            "hpln_centers": bin_hpln, "hplt_centers": bin_hplt}

# -----------------------
# Fast Median Binning
# -----------------------
def per_bin_median(vx, vy, vv, x_bins, y_bins):
    """Highly optimized pure-NumPy spatial median filter."""
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

# -----------------------
# Core Processing
# -----------------------
def process_files(fits_files):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Initializing grid using {os.path.basename(fits_files[0])}...")
    grid = build_global_grid(fits_files[0], BIN_SIZE_DEG)
    
    for p in fits_files:
        data, t, wcs_solar, _, header = load_cim_data(p)
        if data is None: continue

        print(f"Processing: {os.path.basename(p)}")
        
        # Extract Coordinates
        hpln, hplt = wcs_solar.pixel_to_world_values(grid["x_idx"].ravel(), grid["y_idx"].ravel())
        
        # Convert raw data to S10
        flat_s10 = data.ravel() / S10_COEFF
        
        # 1. Apply spatial median filter
        binned_cim = per_bin_median(hpln, hplt, flat_s10, grid["x_bins"], grid["y_bins"])
        
        # Flatten arrays for 1D output writing
        res_s10 = binned_cim.ravel()
        res_ra, res_dec = grid["wcs_radec"].pixel_to_world_values(
            *grid["wcs_solar"].world_to_pixel_values(grid["hpln_centers"].ravel(), grid["hplt_centers"].ravel())
        )
        
        # 2. Filter out edge padding, exact zeros, and invalid math
        valid_mask = (
            np.isfinite(res_s10) & 
            (res_s10 < S10_MAX) & 
            (res_s10 > S10_MIN) &
            (res_s10 != 0.0) &         # Drops the black padding around the FOV
            np.isfinite(res_ra) & 
            np.isfinite(res_dec) &
            (res_ra >= 0.0) &          # Boundary enforcement to prevent math crashes
            (res_ra <= 360.0) &        
            (res_dec >= -90.0) &       
            (res_dec <= 90.0)          
        )
        
        ts = t.strftime("%Y%m%d%H%M%S")
        out_path = os.path.join(OUT_DIR, f"PUNCH_L3_CIM_{ts}_BINNED.txt")
        flat_valid = valid_mask.ravel()
        
        with open(out_path, "w") as f:
            # Header line: YYYY DOY_Fraction
            f.write(f"{t.to_datetime().year} {t.jd - Time(f'{t.to_datetime().year}-01-01').jd + 1.0:.8f}\n")
            
            # Data lines
            for r, d, b in zip(res_ra[flat_valid], res_dec[flat_valid], res_s10[flat_valid]):
                f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {t.to_datetime().isoformat()}\n")

if __name__ == "__main__":
    # You can pass files via command line args, or it will grab all FITS in the dir
    files = sorted(glob.glob("*.fits")) if len(sys.argv) == 1 else sys.argv[1:]
    
    if not files:
        print("No FITS files found in the current directory.")
        sys.exit(1)
        
    process_files(files)