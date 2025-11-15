#!/usr/bin/env python3
"""
bin_punch_l3_to_1deg.py

Takes a PUNCH L3 mosaic FITS file and produces a coarser image
where each pixel represents ~1° x 1° in the original helioprojective
(HPLN/HPLT) coordinates, using a median within each block to reduce starlight.

Usage:
    python bin_punch_l2_to_1deg.py input.fits output_1deg.fits
"""

import sys
import numpy as np
from astropy.io import fits

def bin_l3_to_1deg(input_fits, output_fits):
    # -----------------------------------------------------------
    # 1. Open the FITS and get the L3 mosaic image + header
    # -----------------------------------------------------------
    hdul = fits.open(input_fits)
    # For PUNCH L3, the image is usually in the first extension (BINTABLE with tile compression)
    hdu = hdul[1] if len(hdul) > 1 else hdul[0]

    # Astropy will transparently decompress the image data from a tile-compressed BINTABLE
    data = hdu.data.astype(np.float64)  # shape ~ (4096, 4096)
    hdr = hdu.header

    # -----------------------------------------------------------
    # 2. Read WCS scale: CDELT1, CDELT2 (deg per pixel)
    #    We assume:
    #      - helioprojective HPLN/HPLT grid
    #      - CDELT1 ≈ CDELT2
    #      - no rotation (CROTA = 0, no PC matrix for this WCS)
    # -----------------------------------------------------------
    cdelt1 = float(hdr.get("CDELT1", 0.0))
    cdelt2 = float(hdr.get("CDELT2", 0.0))

    if cdelt1 == 0.0 or cdelt2 == 0.0:
        raise RuntimeError("CDELT1 or CDELT2 missing/zero in header; cannot infer angular scale.")

    # pixels per degree along each axis
    # px_per_deg_x = 1 / |CDELT1|
    # px_per_deg_y = 1 / |CDELT2|
    px_per_deg_x = 1.0 / abs(cdelt1)
    px_per_deg_y = 1.0 / abs(cdelt2)

    # Block (tile) size in pixels, rounded to nearest integer
    block_x = int(round(px_per_deg_x))
    block_y = int(round(px_per_deg_y))

    print(f"CDELT1 = {cdelt1:.6f} deg/pix, CDELT2 = {cdelt2:.6f} deg/pix")
    print(f"Using block size ~1° → {block_x} x {block_y} pixels per block")

    ny, nx = data.shape  # ny = number of rows (y), nx = number of columns (x)
    print(f"Input image size: {nx} x {ny} pixels")

    # -----------------------------------------------------------
    # 3. Trim the image so its size is an integer multiple of the block size
    #    (we drop a small strip at the edges if needed)
    # -----------------------------------------------------------
    nx_trim = (nx // block_x) * block_x
    ny_trim = (ny // block_y) * block_y

    if nx_trim != nx or ny_trim != ny:
        print(f"Trimming image from {nx}x{ny} → {nx_trim}x{ny_trim} to fit whole blocks")

    data_trim = data[:ny_trim, :nx_trim]

    # -----------------------------------------------------------
    # 4. Reshape into blocks and take the median in each block
    #
    #    We reshape:
    #      (ny_trim, nx_trim) -> (n_blocks_y, block_y, n_blocks_x, block_x)
    #    then take the median along the block_y and block_x axes.
    # -----------------------------------------------------------
    n_blocks_x = nx_trim // block_x
    n_blocks_y = ny_trim // block_y

    # Reshape into 4D: (n_blocks_y, block_y, n_blocks_x, block_x)
    blocks = data_trim.reshape(n_blocks_y, block_y, n_blocks_x, block_x)

    # Take median over the block pixels (axes 1 and 3)
    binned = np.nanmedian(blocks, axis=(1, 3))

    print(f"Output binned image size: {n_blocks_x} x {n_blocks_y} blocks (~1° x 1° each)")

    # -----------------------------------------------------------
    # 5. Build a simple output header with approximate WCS
    #    We downscale the original WCS:
    #      - new CDELT = old CDELT * block_size  (~ 1°)
    #      - new CRPIX scaled by block_size
    #    This is approximate but good enough as a rough product.
    # -----------------------------------------------------------
    new_hdr = fits.Header()

    # Copy over some basic metadata (optional – you can add more keys if useful)
    for key in ["PROJECT", "TITLE", "LEVEL", "OBSTYPE", "TYPECODE", "OBSCODE",
                "INSTRUME", "TELESCOP", "OBSRVTRY", "BUNIT"]:
        if key in hdr:
            new_hdr[key] = hdr[key]

    # Approximate WCS for helioprojective coordinates:
    # We keep CTYPE/CUNIT as original
    new_hdr["WCSAXES"] = 2
    new_hdr["CTYPE1"] = hdr.get("CTYPE1", "HPLN-ARC")
    new_hdr["CTYPE2"] = hdr.get("CTYPE2", "HPLT-ARC")
    new_hdr["CUNIT1"] = hdr.get("CUNIT1", "deg")
    new_hdr["CUNIT2"] = hdr.get("CUNIT2", "deg")

    # New pixel scale: ~1° per pixel (old scale * block_size)
    new_hdr["CDELT1"] = cdelt1 * block_x
    new_hdr["CDELT2"] = cdelt2 * block_y

    # Reference coordinate values (keep same CRVAL as original)
    crval1 = float(hdr.get("CRVAL1", 0.0))
    crval2 = float(hdr.get("CRVAL2", 0.0))
    new_hdr["CRVAL1"] = crval1
    new_hdr["CRVAL2"] = crval2

    # Reference pixel (scale old CRPIX down to new grid)
    # Old: world = CRVAL + (i_old - CRPIX_old)*CDELT_old
    # New: world = CRVAL + (i_new - CRPIX_new)*CDELT_new
    # Approx mapping: i_new ≈ (i_old - 0.5)/block + 0.5
    crpix1_old = float(hdr.get("CRPIX1", (nx + 1) / 2.0))
    crpix2_old = float(hdr.get("CRPIX2", (ny + 1) / 2.0))
    crpix1_new = (crpix1_old - 0.5) / block_x + 0.5
    crpix2_new = (crpix2_old - 0.5) / block_y + 0.5
    new_hdr["CRPIX1"] = crpix1_new
    new_hdr["CRPIX2"] = crpix2_new

    # Set image size keywords
    new_hdr["NAXIS"] = 2
    new_hdr["NAXIS1"] = n_blocks_x
    new_hdr["NAXIS2"] = n_blocks_y

    # You can also propagate DATE-OBS, TIMESYS, etc., if useful:
    for key in ["TIMESYS", "DATE-OBS", "DATE-BEG", "DATE-END", "RSUN_ARC",
                "RSUN_REF", "DSUN_OBS", "HGLT_OBS", "HGLN_OBS"]:
        if key in hdr:
            new_hdr[key] = hdr[key]

    # -----------------------------------------------------------
    # 6. Write output FITS
    # -----------------------------------------------------------
    hdu_out = fits.PrimaryHDU(data=binned.astype(np.float32), header=new_hdr)
    hdu_out.writeto(output_fits, overwrite=True)
    hdul.close()

    print(f"Written binned image to: {output_fits}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bin_punch_l3_to_1deg.py input.fits output_1deg.fits")
        sys.exit(1)

    bin_l3_to_1deg(sys.argv[1], sys.argv[2])