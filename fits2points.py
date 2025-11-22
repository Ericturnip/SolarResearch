import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
import sys

def punch_fits_to_txt(input_fits, output_txt, cam_id="L3"):
    # --- 1. CONSTANTS ---
    # S10 Coefficient only. 
    # We assume data is already MSB (Mean Solar Brightness) based on your previous values.
    S10_COEFF = 4.5e-16 

    print(f"Reading {input_fits}...")
    
    with fits.open(input_fits) as hdul:
        # Check index 0 or 1 for data
        if hdul[0].data is not None:
            data = hdul[0].data
            header = hdul[0].header
        else:
            data = hdul[1].data
            header = hdul[1].header

    # Handle dimensions
    data = data.squeeze() 

    # --- 2. GENERATE COORDINATES (RA/DEC) ---
    wcs = WCS(header)
    height, width = data.shape
    y_indices, x_indices = np.indices((height, width))
    
    # Convert pixels to RA/DEC
    ra_map, dec_map = wcs.wcs_pix2world(x_indices, y_indices, 0)

    # --- 3. APPLY CONVERSION ---
    # logic: Value_S10 = Value_MSB / 4.5e-16
    s10_data = data / S10_COEFF

    # Flatten arrays
    flat_ra = ra_map.flatten()
    flat_dec = dec_map.flatten()
    flat_data = s10_data.flatten()

    # --- 4. FILTER AND FORMAT ---
    # Filter out 0 and NaN
    valid_mask = (flat_data != 0) & (~np.isnan(flat_data))
    
    clean_ra = flat_ra[valid_mask]
    clean_dec = flat_dec[valid_mask]
    clean_data = flat_data[valid_mask]

    # Parse Date for Header (YYYY DOY.fraction)
    try:
        obs_date = header.get('DATE-OBS')
        t = Time(obs_date)
        jan1 = Time(f"{t.ymdh[0]}-01-01 00:00:00")
        doy = t.jd - jan1.jd + 1.0
        date_string = f"{t.ymdh[0]} {doy:.8f}"
    except Exception as e:
        print(f"Date warning: {e}")
        date_string = "0000 000.00000000"

    # --- 5. WRITE OUTPUT ---
    print(f"Writing {len(clean_data)} points to {output_txt}...")
    
    with open(output_txt, 'w') as f:
        f.write(f"{date_string}\n")
        # Format: ID  RA  DEC  BRIGHTNESS
        # 6.2f ensures you see 41.65, etc.
        for r, d, b in zip(clean_ra, clean_dec, clean_data):
            f.write(f"{cam_id}  {r:6.2f} {d:6.2f}  {b:6.2f}\n")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = input_file.replace(".fits", ".txt")
        punch_fits_to_txt(input_file, output_file)
    else:
        print("Usage: python3 script.py <your_file.fits>")