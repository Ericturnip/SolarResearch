import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
from scipy.ndimage import grey_closing
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
        return t.strftime('%Y%m%d%H%M%S')
    return "00000000000000"

def load_fits_data(input_fits):
    """Load data and WCS."""
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
        date_obs = header.get('DATE-OBS')
        if not date_obs: return None, None, None, None, None
        t = Time(date_obs, format='isot', scale='utc')
        wcs_solar = WCS(header)
        try:
            wcs_radec = WCS(header, key='A')
        except Exception:
            wcs_radec = None
        return data, t, wcs_solar, wcs_radec, header
    except Exception as e:
        print(f"Error processing {input_fits}: {e}")
        return None, None, None, None, None

def heal_dark_seams(image_data, seam_size=3):
    """Applies morphological closing to fill dark cracks."""
    temp_filled = image_data.copy()
    mask_nan = np.isnan(temp_filled)
    # If image is totally empty, return as is
    if np.all(mask_nan): return image_data
    
    # Fill NaNs with median to allow filter to work
    global_median = np.nanmedian(temp_filled)
    temp_filled[mask_nan] = global_median
    
    # Apply closing
    healed = grey_closing(temp_filled, size=(seam_size, seam_size))
    
    # Restore NaNs
    healed[mask_nan] = np.nan
    return healed

def process_punch_l3s_percentile_stack(input_fits_list, bin_size_deg=1.0, percentile=25):
    # --- 1. CONSTANTS ---
    S10_COEFF = 4.5e-16 

    if not input_fits_list:
        print("Error: Input FITS list is empty.")
        return

    # --- 2. INITIALIZE GRID ---
    print(f"Initializing percentile process ({percentile}%) using {os.path.basename(input_fits_list[0])}...")
    data_init, t_init, wcs_solar_init, wcs_radec_init, header_init = load_fits_data(input_fits_list[0])
    
    if data_init is None: return
        
    height, width = data_init.shape
    y_idx, x_idx = np.indices((height, width))
    flat_hpln, flat_hplt = wcs_solar_init.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())

    min_x, max_x = np.min(flat_hpln), np.max(flat_hpln)
    min_y, max_y = np.min(flat_hplt), np.max(flat_hplt)

    x_bins = np.arange(np.floor(min_x), np.ceil(max_x) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(min_y), np.ceil(max_y) + bin_size_deg, bin_size_deg)

    num_x_bins = len(x_bins) - 1
    num_y_bins = len(y_bins) - 1
    
    # --- 3. CREATE STACK ---
    n_files = len(input_fits_list)
    print(f"Allocating stack memory: ({n_files}, {num_y_bins}, {num_x_bins})")
    
    s10_stack = np.full((n_files, num_y_bins, num_x_bins), np.nan, dtype=np.float32)
    time_stack = np.full((n_files, num_y_bins, num_x_bins), "", dtype='<U30')

    bin_hpln_centers = binned_statistic_2d(flat_hpln, flat_hplt, flat_hpln, statistic='mean', bins=[x_bins, y_bins]).statistic.T
    bin_hplt_centers = binned_statistic_2d(flat_hpln, flat_hplt, flat_hplt, statistic='mean', bins=[x_bins, y_bins]).statistic.T

    start_timestamp = get_timestamp_from_header(header_init)
    end_timestamp = start_timestamp

    # --- 4. POPULATE STACK ---
    print(f"Loading {n_files} files into stack...")

    for i, input_fits in enumerate(input_fits_list):
        print(f"  [{i+1}/{n_files}] Processing {os.path.basename(input_fits)}...")
        data, t, wcs_curr, _, header_curr = load_fits_data(input_fits)
        if data is None: continue
        
        if i == n_files - 1: end_timestamp = get_timestamp_from_header(header_curr)

        flat_data = data.flatten()
        curr_hpln, curr_hplt = wcs_curr.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())
        flat_s10 = (flat_data / S10_COEFF)
        
        # Filter: Exclude NaNs, Saturated, and EXACT ZEROS
        valid_pixel_mask = (~np.isnan(flat_s10)) & (flat_s10 < 500) & (flat_s10 > -50) & (flat_s10 != 0)
        
        img_binned = binned_statistic_2d(
            curr_hpln[valid_pixel_mask], 
            curr_hplt[valid_pixel_mask], 
            flat_s10[valid_pixel_mask], 
            statistic='min', 
            bins=[x_bins, y_bins]
        ).statistic.T
        
        s10_stack[i, :, :] = img_binned
        valid_bins_mask = ~np.isnan(img_binned)
        time_stack[i][valid_bins_mask] = t.to_datetime().isoformat()

    # --- 5. COMPUTE PERCENTILE ---
    print(f"Calculating {percentile}th Percentile Stack...")
    
    # np.nanpercentile ignores NaNs but returns NaN if column is all-empty
    final_s10 = np.nanpercentile(s10_stack, percentile, axis=0)

    # --- 6. SPATIAL HEALING ---
    final_s10 = heal_dark_seams(final_s10, seam_size=3)

    # --- 7. MATCH TIMESTAMPS (CRASH FIX HERE) ---
    print("Matching timestamps...")
    
    # Calculate distance from final value
    dist_from_final = np.abs(s10_stack - final_s10)
    
    # FIX: Replace NaNs with Infinity so argmin doesn't choose them
    # AND to prevent "All-NaN slice" error
    dist_from_final[np.isnan(dist_from_final)] = np.inf
    
    # Use standard argmin (safe on all-Infs, returns 0)
    closest_indices = np.argmin(dist_from_final, axis=0)
    
    yy, xx = np.indices((num_y_bins, num_x_bins))
    final_time = time_stack[closest_indices, yy, xx]

    # --- 8. SAVE ---
    res_s10 = final_s10.flatten()
    res_hpln = bin_hpln_centers.flatten()
    res_hplt = bin_hplt_centers.flatten()
    res_time = final_time.flatten()

    print("Converting to RA/DEC...")
    target_pix_x, target_pix_y = wcs_solar_init.world_to_pixel_values(res_hpln, res_hplt)
    res_ra, res_dec = wcs_radec_init.pixel_to_world_values(target_pix_x, target_pix_y)

    valid_mask = (~np.isinf(res_s10)) & (~np.isnan(res_s10)) & (res_time != "") & (~np.isnan(res_ra))
    clean_ra = res_ra[valid_mask]
    clean_dec = res_dec[valid_mask]
    clean_s10 = res_s10[valid_mask]
    clean_time = res_time[valid_mask]
    
    t_dt = t_init.to_datetime()
    jan1_dt = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1_dt).jd
    doy_fraction = t_init.jd - jan1_jd + 1.0
    header_date_string = f"{t_dt.year} {doy_fraction:.8f}"

    output_filename = f"PUNCH_L3_CIM_RANGE_{start_timestamp}_{end_timestamp}_p{percentile}_healed.txt"

    print(f"Writing {len(clean_s10)} points to {output_filename}...")
    with open(output_filename, 'w') as f:
        f.write(f"{header_date_string}\n")
        for r, d, b, tm in zip(clean_ra, clean_dec, clean_s10, clean_time):
            f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {tm}\n")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_files = sys.argv[1:]
        process_punch_l3s_percentile_stack(input_files)
    else:
        print("Scanning current directory for .fits files...")
        fits_files = sorted(glob.glob("*.fits"))
        if fits_files:
            process_punch_l3s_percentile_stack(fits_files)
        else:
            print("No .fits files found.")