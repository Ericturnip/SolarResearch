# PUNCH Pipeline v4 Scaffold

This is a clean scaffold for PUNCH processing. Product-specific data interpretation is intentionally isolated in adapters; the processing logic is generic.

## Core design

- **Adapters** convert any product type into `LayerFrame` objects: named 2D layers + WCS + timestamp + metadata.
- **Median filter** means: map one image/layer into 1°×1° sky bins and take the median in each bin.
- **Composite** means: combine already-binned maps across a time window using nanmedian/min/p30/etc.
- **TXT writer** writes tomography-style ASCII rows from binned maps.
- **Second pass** is a separate stage with placeholder hooks to be filled from the existing code later.

## Processing stages

### 1. Single-image median binning
Input: one FITS or one already-loaded frame.
Output: one binned TXT file.

```bash
python scripts/run_median_filter.py \
  --input path/to/file.fits \
  --product CIM \
  --layer brightness \
  --output-dir outputs/median \
  --bin-size-deg 1.0
```

### 2. One-hour composite
Input: FITS files or TXT files.
Output: one binned TXT composite.

```bash
python scripts/run_hourly_composite.py \
  --input-glob 'data/*.fits' \
  --product CIM \
  --layer brightness \
  --output-dir outputs/composite \
  --composite-method nanmedian
```

### 3. Second pass
Input: first-pass TXT/composite outputs.
Output: cleaned second-pass TXT outputs.

```bash
python scripts/run_second_pass.py \
  --input-glob 'outputs/composite/*.txt' \
  --output-dir outputs/second_pass
```

## Product adapters

Files exist for CIM, CTM, PIM, PTM, PAM, CAM, but they intentionally raise `NotImplementedError` until filled.

## Important defaults

- TXT outputs default to `outputs/`, never `/tmp`.
- Zero values are treated as missing by default before TXT/min/composite output.
- Native brightness can be converted to S10 with `S10_COEFF = 4.5e-16`.
- Empty bins are omitted.
