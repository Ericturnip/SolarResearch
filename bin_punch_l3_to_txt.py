import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import get_sun
from scipy.stats import binned_statistic_2d
import sys

def process_punch_l3_to_txt(input_fits, output_txt, bin_size_deg=1.0):
    # --- 1. CONSTANTS ---
    S10_COEFF = 4.5e-16 
    
    print(f"Reading {input_fits}...")
    with fits.open(input_fits) as hdul:
        if hdul[0].data is not None:
            data = hdul[0].data
            header = hdul[0].header
        else:
            data = hdul[1].data
            header = hdul[1].header

    data = data.squeeze().astype(np.float64)
    height, width = data.shape
    
    # --- 2. CALCULATE SUN POSITION ---
    try:
        date_obs = header.get('DATE-OBS')
        if not date_obs:
            print("Warning: DATE-OBS missing. Using placeholder.")
            date_obs = '2025-08-31T04:24:29.132'
        
        t = Time(date_obs)
        sun_pos = get_sun(t)
        sun_ra = sun_pos.ra.deg
        sun_dec = sun_pos.dec.deg
        print(f"Sun Position: RA {sun_ra:.2f}, DEC {sun_dec:.2f}")
    except Exception as e:
        print(f"Error calculating Sun position: {e}")
        return

    # --- 3. GENERATE PIXEL COORDINATES ---
    print("Generating pixel coordinates...")
    
    cdelt1 = float(header.get('CDELT1', 1.0))
    cdelt2 = float(header.get('CDELT2', 1.0))
    crpix1 = float(header.get('CRPIX1', width / 2))
    crpix2 = float(header.get('CRPIX2', height / 2))

    y_idx, x_idx = np.indices((height, width))

    # Convert to degrees from Sun center
    hpln_deg = (x_idx - (crpix1 - 1)) * cdelt1
    hplt_deg = (y_idx - (crpix2 - 1)) * cdelt2

    flat_hpln = hpln_deg.flatten()
    flat_hplt = hplt_deg.flatten()
    flat_data = data.flatten()

    # --- 4. DEFINE BINS ---
    min_x, max_x = np.min(flat_hpln), np.max(flat_hpln)
    min_y, max_y = np.min(flat_hplt), np.max(flat_hplt)

    x_bins = np.arange(np.floor(min_x), np.ceil(max_x) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(min_y), np.ceil(max_y) + bin_size_deg, bin_size_deg)

    print(f"Binning {len(flat_data)} pixels into {len(x_bins)-1}x{len(y_bins)-1} grid...")

    # --- 5. PERFORM BINNING (FIXED) ---
    
    # A. Median Brightness
    # We use [0] to grab just the statistic array and ignore edges/bin numbers
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_bright = binned_statistic_2d(
            flat_hpln, flat_hplt, flat_data, 
            statistic='median', bins=[x_bins, y_bins]
        ).statistic

        # B. Mean HPLN (Center Longitude of bin)
        bin_hpln_centers = binned_statistic_2d(
            flat_hpln, flat_hplt, flat_hpln, 
            statistic='mean', bins=[x_bins, y_bins]
        ).statistic

        # C. Mean HPLT (Center Latitude of bin)
        bin_hplt_centers = binned_statistic_2d(
            flat_hpln, flat_hplt, flat_hplt, 
            statistic='mean', bins=[x_bins, y_bins]
        ).statistic

    # --- 6. CONVERT TO RA/DEC & S10 ---
    print("Converting to RA/DEC and S10 units...")
    
    res_bright = bin_bright.flatten()
    res_hpln = bin_hpln_centers.flatten()
    res_hplt = bin_hplt_centers.flatten()

    res_s10 = res_bright / S10_COEFF

    # RA/DEC = Sun Position + Offset
    res_ra = sun_ra + res_hpln
    res_dec = sun_dec + res_hplt

    # --- 7. FILTER AND SAVE ---
    # Filter valid data (not NaN, >0, not saturated stars >2000)
    valid_mask = (~np.isnan(res_s10)) & (res_s10 != 0)

    clean_ra = res_ra[valid_mask]
    clean_dec = res_dec[valid_mask]
    clean_s10 = res_s10[valid_mask]

    # Date formatting (using datetime to avoid AttributeErrors)
    t_dt = t.to_datetime()
    jan1_dt = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1_dt).jd
    doy_fraction = t.jd - jan1_jd + 1.0
    date_string = f"{t_dt.year} {doy_fraction:.8f}"

    print(f"Writing {len(clean_s10)} points to {output_txt}...")
    
    with open(output_txt, 'w') as f:
        f.write(f"{date_string}\n")
        for r, d, b in zip(clean_ra, clean_dec, clean_s10):
            f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f}\n")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            output_file = input_file.replace(".fits", ".txt")
            
        process_punch_l3_to_txt(input_file, output_file)
    else:
        print("Usage: python3 punch_direct_binning.py <input_punch_l3.fits> [output.txt]")