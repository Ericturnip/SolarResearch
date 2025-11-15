#!/usr/bin/env python3
"""
remove_stars_with_1deg_bg.py

Given:
  1) A full-resolution PUNCH L3 (or L2) mosaic FITS file,
  2) A 1° x 1° binned median map produced from that mosaic,

this script:
  - expands the coarse 1° map back to (almost) the original resolution,
  - computes the residual (original - background),
  - detects star pixels as high-sigma outliers in the residual,
  - replaces those star pixels with the background value,
  - writes a star-cleaned FITS image.

Usage:
    python remove_stars_with_1deg_bg.py full_res.fits coarse_1deg.fits output_clean.fits

Notes:
  - Assumes the coarse map covers the central region of the image and
    that its size is an integer factor of the full-res image size
    (as in the binning script we wrote earlier).
"""

import sys
import numpy as np
from astropy.io import fits

def robust_sigma(residual):
    """
    Estimate a robust sigma using Median Absolute Deviation (MAD).
    sigma ≈ 1.4826 * median(|residual - median(residual)|)

    If MAD = 0 (pathological), fall back to standard deviation.
    """
    med = np.nanmedian(residual)
    mad = np.nanmedian(np.abs(residual - med))
    if mad == 0.0:
        return np.nanstd(residual)
    return 1.4826 * mad

def remove_stars(full_res_fits, coarse_fits, output_fits, sigma_thresh=5.0):
    # -----------------------------------------------------------
    # 1. Load full-resolution mosaic
    # -----------------------------------------------------------
    hdul_full = fits.open(full_res_fits)
    # As before: PUNCH products often put the image in extension 1 (BINTABLE), but
    # astropy .data will give you a 2D array for compressed images as well.
    hdu_full = hdul_full[1] if len(hdul_full) > 1 else hdul_full[0]
    data_full = hdu_full.data.astype(np.float64)
    hdr_full = hdu_full.header

    ny_full, nx_full = data_full.shape
    print(f"Full-res image size: {nx_full} x {ny_full}")

    # -----------------------------------------------------------
    # 2. Load the coarse 1° x 1° binned map
    # -----------------------------------------------------------
    hdul_coarse = fits.open(coarse_fits)
    hdu_coarse = hdul_coarse[0]  # we wrote it as a PrimaryHDU
    data_coarse = hdu_coarse.data.astype(np.float64)
    hdr_coarse = hdu_coarse.header

    ny_coarse, nx_coarse = data_coarse.shape
    print(f"Coarse 1° map size: {nx_coarse} x {ny_coarse}")

    # -----------------------------------------------------------
    # 3. Determine block sizes (pixels per 1° bin) from shapes
    #
    #    We assume the coarse map came from tiling the central region:
    #      nx_trim = block_x * nx_coarse
    #      ny_trim = block_y * ny_coarse
    #    and nx_trim <= nx_full, ny_trim <= ny_full.
    # -----------------------------------------------------------
    block_x = nx_full // nx_coarse
    block_y = ny_full // ny_coarse

    if block_x <= 0 or block_y <= 0:
        raise RuntimeError("Coarse map appears larger than full-res image; check inputs.")

    nx_trim = block_x * nx_coarse
    ny_trim = block_y * ny_coarse

    if nx_trim != nx_full or ny_trim != ny_full:
        print(f"Trimming full-res image from {nx_full}x{ny_full} → {nx_trim}x{ny_trim} "
              f"to match tiled coarse map region.")

    data_trim = data_full[:ny_trim, :nx_trim]

    print(f"Block size inferred: {block_x} x {block_y} pixels per coarse pixel.")

    # -----------------------------------------------------------
    # 4. Expand the coarse 1° map back to the trimmed resolution
    #    using simple nearest-neighbour tiling:
    #       upsampled_bg(y, x) = data_coarse[y//block_y, x//block_x]
    #    Implemented via np.repeat along each axis.
    # -----------------------------------------------------------
    bg_upsampled = np.repeat(
        np.repeat(data_coarse, block_y, axis=0),
        block_x, axis=1
    )

    # Ensure shapes match
    bg_upsampled = bg_upsampled[:ny_trim, :nx_trim]
    assert bg_upsampled.shape == data_trim.shape

    # -----------------------------------------------------------
    # 5. Compute residual and robust sigma
    # -----------------------------------------------------------
    residual = data_trim - bg_upsampled

    sigma = robust_sigma(residual)
    print(f"Robust sigma estimate of residual: {sigma:.3e}")

    if not np.isfinite(sigma) or sigma == 0.0:
        raise RuntimeError("Non-finite or zero sigma; residual looks pathological.")

    # -----------------------------------------------------------
    # 6. Detect star pixels as high-sigma positive outliers
    #
    #    We look for R > sigma_thresh * sigma.
    #    You can tune sigma_thresh (default 5.0) depending on how aggressive you want.
    # -----------------------------------------------------------
    threshold = sigma_thresh * sigma
    print(f"Using sigma threshold: {sigma_thresh:.1f}σ → {threshold:.3e}")

    star_mask = residual > threshold
    n_star_pixels = np.count_nonzero(star_mask)
    frac = n_star_pixels / star_mask.size * 100.0
    print(f"Star mask: {n_star_pixels} pixels flagged ({frac:.3f} % of trimmed image).")

    # Optional: you could dilate the mask to include PSF wings (requires scipy):
    # from scipy.ndimage import binary_dilation
    # star_mask = binary_dilation(star_mask, iterations=1)

    # -----------------------------------------------------------
    # 7. Build cleaned image: replace star pixels with background
    # -----------------------------------------------------------
    cleaned_trim = data_trim.copy()
    cleaned_trim[star_mask] = bg_upsampled[star_mask]

    # -----------------------------------------------------------
    # 8. Build an output header and write result
    #
    #    For simplicity, we:
    #      - copy the full header,
    #      - update NAXIS1/2 to trimmed shape,
    #      - write as a new primary HDU (uncompressed).
    # -----------------------------------------------------------
    new_hdr = hdr_full.copy()

    # Update size keywords to match trimmed region
    new_hdr["NAXIS"] = 2
    new_hdr["NAXIS1"] = nx_trim
    new_hdr["NAXIS2"] = ny_trim

    # If the original had WCS, the reference pixel CRPIX might technically shift
    # if we trimmed off non-zero margins. Here we assumed trimming only at far edges,
    # i.e., CRPIX still within the kept region. If you ever trim around the center,
    # you should adjust CRPIX accordingly.

    hdu_out = fits.PrimaryHDU(data=cleaned_trim.astype(np.float32), header=new_hdr)
    hdu_out.writeto(output_fits, overwrite=True)

    hdul_full.close()
    hdul_coarse.close()

    print(f"Star-cleaned image written to: {output_fits}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python remove_stars_with_1deg_bg.py full_res.fits coarse_1deg.fits output_clean.fits")
        sys.exit(1)

    full_res_fits = sys.argv[1]
    coarse_fits = sys.argv[2]
    output_fits = sys.argv[3]

    remove_stars(full_res_fits, coarse_fits, output_fits, sigma_thresh=5.0)