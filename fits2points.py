import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord
import astropy.units as u
import sys

def generate_punch_text_file(input_fits, output_txt, cam_id="L3"):
    # --- 1. CONSTANTS ---
    S10_COEFF = 4.5e-16 

    print(f"Reading {input_fits}...")
    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # Handle dimensions
    data = data.squeeze()
    height, width = data.shape

    # --- 2. RECOVER TIME AND SUN POSITION ---
    # We need the Sun's location to anchor our RA/DEC grid
    try:
        date_obs = header.get('DATE-OBS')
        if not date_obs:
            # Fallback if header is missing date (using the one from your example)
            print("Warning: DATE-OBS missing. Using default.")
            date_obs = '2025-08-31T04:24:29.132'
        
        t = Time(date_obs)
        
        # Calculate Sun's Position (RA/DEC) for this time
        # This gives us the 'center' of the helioprojective view in ICRS
        sun_pos = get_sun(t)
        sun_ra = sun_pos.ra.deg
        sun_dec = sun_pos.dec.deg
        print(f"Sun Position calculated: RA {sun_ra:.2f}, DEC {sun_dec:.2f}")

    except Exception as e:
        print(f"Critical Error calculating Sun position: {e}")
        return

    # --- 3. MANUAL COORDINATE CALCULATION ---
    # Since the full WCS is broken, we compute offsets manually using CDELT
    # RA = Sun_RA + (Pixel_X - Reference_Pixel_X) * Degrees_Per_Pixel
    
    # Get Pixel Scale (degrees per pixel)
    cdelt1 = header.get('CDELT1', 1.0) # Default to 1 if missing
    cdelt2 = header.get('CDELT2', 1.0)
    
    # Get Reference Pixel (Center of Sun in image)
    crpix1 = header.get('CRPIX1', width / 2)
    crpix2 = header.get('CRPIX2', height / 2)

    print(f"Mapping grid using scale: {cdelt1:.4f} deg/pix")

    # Create Grid
    x_pixels, y_pixels = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply linear transformation (simplified WCS)
    # Note: standard FITS is 1-based, numpy is 0-based. 
    # We adjust (x - (crpix - 1)) to align correctly.
    ra_map = sun_ra + (x_pixels - (crpix1 - 1)) * cdelt1
    dec_map = sun_dec + (y_pixels - (crpix2 - 1)) * cdelt2

    # --- 4. CONVERT BRIGHTNESS ---
    s10_data = data / S10_COEFF

    # Flatten arrays
    flat_ra = ra_map.flatten()
    flat_dec = dec_map.flatten()
    flat_data = s10_data.flatten()

    # --- 5. FILTERING ---
    # Remove NaNs, 0s, and Saturated Stars (> 2000)
    valid_mask = (flat_data > 0) & (~np.isnan(flat_data)) & (flat_data < 2000)
    
    clean_ra = flat_ra[valid_mask]
    clean_dec = flat_dec[valid_mask]
    clean_data = flat_data[valid_mask]

    # --- 6. FORMAT DATE ---
    # Calculate fractional DOY manually using standard Python datetime to avoid attribute errors
    t_dt = t.to_datetime()
    jan1_dt = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1_dt).jd
    doy_fraction = t.jd - jan1_jd + 1.0
    date_string = f"{t_dt.year} {doy_fraction:.8f}"

    # --- 7. WRITE OUTPUT ---
    print(f"Writing {len(clean_data)} points to {output_txt}...")
    
    with open(output_txt, 'w') as f:
        f.write(f"{date_string}\n")
        for r, d, b in zip(clean_ra, clean_dec, clean_data):
            f.write(f"{cam_id}  {r:6.2f} {d:6.2f}  {b:6.2f}\n")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = input_file.replace(".fits", ".txt")
        generate_punch_text_file(input_file, output_file)
    else:
        print("Usage: python3 punch_final_converter.py <input_1deg.fits>")