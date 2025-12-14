import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
import sys
import os
import glob
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def get_timestamp_from_header(header):
    """Extracts a clean timestamp string (YYYYMMDDHHMMSS) from a header."""
    date_obs = header.get('DATE-OBS')
    if date_obs:
        t = Time(date_obs, format='isot', scale='utc')
        # Format: YYYYMMDDHHMMSS
        return t.strftime('%Y%m%d%H%M%S')
    return "00000000000000"

def load_fits_data(input_fits):
    """
    Load data and WCS. Returns data, time object, solar WCS, and RA/DEC WCS.
    """
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

        data = data.squeeze().astype(np.float64)
        
        # Get Time
        date_obs = header.get('DATE-OBS')
        if not date_obs:
            return None, None, None, None, None
        t = Time(date_obs, format='isot', scale='utc')

        # Load WCS
        wcs_solar = WCS(header)
        try:
            wcs_radec = WCS(header, key='A')
        except Exception:
            wcs_radec = None

        return data, t, wcs_solar, wcs_radec, header

    except Exception as e:
        print(f"Error processing {input_fits}: {e}")
        return None, None, None, None, None

def process_punch_l3s_min_to_txt(input_fits_list, bin_size_deg=1.0):
    # --- 1. CONSTANTS ---
    S10_COEFF = 4.5e-16 

    if not input_fits_list:
        print("Error: Input FITS list is empty.")
        return

    # --- 2. INITIALIZE ARRAYS (based on first image) ---
    print(f"Initializing process using {os.path.basename(input_fits_list[0])}...")
    
    data_init, t_init, wcs_solar_init, wcs_radec_init, header_init = load_fits_data(input_fits_list[0])
    
    if data_init is None or wcs_solar_init is None:
        print("Could not initialize. Exiting.")
        return
        
    height, width = data_init.shape

    # Generate Grid Coordinates (HPLN/HPLT)
    y_idx, x_idx = np.indices((height, width))
    flat_hpln, flat_hplt = wcs_solar_init.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())

    # Define Bins
    min_x, max_x = np.min(flat_hpln), np.max(flat_hpln)
    min_y, max_y = np.min(flat_hplt), np.max(flat_hplt)

    x_bins = np.arange(np.floor(min_x), np.ceil(max_x) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(min_y), np.ceil(max_y) + bin_size_deg, bin_size_deg)

    num_x_bins = len(x_bins) - 1
    num_y_bins = len(y_bins) - 1

    # Initialize Storage Arrays (ROWS=Y, COLS=X)
    min_s10_bin = np.full((num_y_bins, num_x_bins), np.inf, dtype=np.float64)
    min_time_bin = np.full((num_y_bins, num_x_bins), "", dtype='<U30')
    
    # Store Bin Centers
    # FIX: Transpose (.T) the result because binned_statistic returns (x,y) but we want (y,x)
    bin_hpln_centers = binned_statistic_2d(
        flat_hpln, flat_hplt, flat_hpln, 
        statistic='mean', bins=[x_bins, y_bins]
    ).statistic.T

    bin_hplt_centers = binned_statistic_2d(
        flat_hpln, flat_hplt, flat_hplt, 
        statistic='mean', bins=[x_bins, y_bins]
    ).statistic.T

    # Time tracking for filename
    start_timestamp = get_timestamp_from_header(header_init)
    end_timestamp = start_timestamp # Will update in loop

    # --- 3. ITERATE AND FIND MINIMUM ---
    print(f"Processing {len(input_fits_list)} FITS files...")

    for i, input_fits in enumerate(input_fits_list):
        print(f"  [{i+1}/{len(input_fits_list)}] Reading {os.path.basename(input_fits)}...")
        
        data, t, wcs_curr, _, header_curr = load_fits_data(input_fits)

        if data is None:
            continue
        
        # Update end timestamp
        if i == len(input_fits_list) - 1:
            end_timestamp = get_timestamp_from_header(header_curr)

        # Process Data
        flat_data = data.flatten()
        curr_hpln, curr_hplt = wcs_curr.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())
        flat_s10 = flat_data / S10_COEFF
        
        # Binning Indices
        x_indices = np.digitize(curr_hpln, x_bins) - 1
        y_indices = np.digitize(curr_hplt, y_bins) - 1
        
        # Filter
        valid_pixel_mask = (x_indices >= 0) & (x_indices < num_x_bins) & \
                           (y_indices >= 0) & (y_indices < num_y_bins) & \
                           (~np.isnan(flat_s10)) & (flat_s10 > 0) & (flat_s10 < 2000)
                           
        valid_s10_vals = flat_s10[valid_pixel_mask]
        
        # Min per bin for this image
        # FIX: Transpose (.T) to match (y,x) shape
        current_img_min_grid = binned_statistic_2d(
            curr_hpln[valid_pixel_mask], 
            curr_hplt[valid_pixel_mask], 
            valid_s10_vals, 
            statistic='min', 
            bins=[x_bins, y_bins]
        ).statistic.T

        # Handle Empty Bins (NaNs)
        current_img_min_grid[np.isnan(current_img_min_grid)] = np.inf
        
        # Update Master
        better_mask = current_img_min_grid < min_s10_bin
        min_s10_bin[better_mask] = current_img_min_grid[better_mask]
        min_time_bin[better_mask] = t.to_datetime().isoformat()

    # --- 4. CONVERT TO RA/DEC ---
    print("Converting grid to RA/DEC...")

    res_s10 = min_s10_bin.flatten()
    res_hpln = bin_hpln_centers.flatten()
    res_hplt = bin_hplt_centers.flatten()
    res_time = min_time_bin.flatten()

    target_pix_x, target_pix_y = wcs_solar_init.world_to_pixel_values(res_hpln, res_hplt)
    res_ra, res_dec = wcs_radec_init.pixel_to_world_values(target_pix_x, target_pix_y)

    valid_mask = (~np.isinf(res_s10)) & (~np.isnan(res_s10)) & (res_time != "") & (~np.isnan(res_ra))

    clean_ra = res_ra[valid_mask]
    clean_dec = res_dec[valid_mask]
    clean_s10 = res_s10[valid_mask]
    clean_time = res_time[valid_mask]
    
    # --- 5. GENERATE HEADER DATE ---
    t_dt = t_init.to_datetime()
    jan1_dt = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1_dt).jd
    doy_fraction = t_init.jd - jan1_jd + 1.0
    header_date_string = f"{t_dt.year} {doy_fraction:.8f}"

    # --- 6. DETERMINE OUTPUT FILENAME ---
    output_filename = f"PUNCH_L3_CIM_RANGE_{start_timestamp}_{end_timestamp}_min_bin.txt"

    # --- 7. SAVE ---
    print(f"Writing {len(clean_s10)} points to {output_filename}...")
    
    with open(output_filename, 'w') as f:
        # 1. Date Header
        f.write(f"{header_date_string}\n")
        
        # 2. Data Lines
        # Formatting: L3 (2 spaces) RA (space) DEC (2 spaces) S10 (padded) TIME
        for r, d, b, tm in zip(clean_ra, clean_dec, clean_s10, clean_time):
            # Using formatting to match your example:
            # L3  179.38 -31.66  248.02 2025-11-10T10:00:29.292
            f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {tm}\n")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User provided specific files
        input_files = sys.argv[1:]
        process_punch_l3s_min_to_txt(input_files)
    else:
        # Auto scan
        print("Scanning current directory for .fits files...")
        fits_files = sorted(glob.glob("*.fits"))
        if fits_files:
            process_punch_l3s_min_to_txt(fits_files)
        else:
            print("No .fits files found.")