# PUNCH Fortran TXT Pipeline

This repo turns PUNCH L3 FITS files into fixed-width TXT maps for the tomography
Fortran code.

The normal run downloads FITS files from NASA, bins each image onto a 1 degree
sky grid, builds one map per UTC hour, and writes rows in the format the Fortran
reader expects.

For p25 hourly maps, the pipeline now uses p25 as a target. It writes the real
input sample closest to that p25 target for each output pixel, with the timestamp
from the same source frame. That keeps brightness and time paired.

## Install

Use a Python environment with `numpy`, `astropy`, `scipy`, `pandas`, and
`matplotlib`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

An existing Anaconda environment is fine if those packages are already there.

## Products

| Product | Default layer | Status |
| --- | --- | --- |
| CTM | `brightness` | Used for production hourly maps. |
| PTM | `Polar_pB` | Used for production hourly maps when NASA has PTM files. |
| CIM | `brightness` | Adapter works for local and downloaded maps. |
| PIM | `Polar_pB` | Computes pB from `Polar_M`, `Polar_Z`, and `Polar_P`. |
| PAM | `Polar_pB` | Same style as PTM. |

CAM is not included because the adapter was still a placeholder.

## One Command For A Month

This wrapper runs CTM, sanitizes the TXT files, then runs PTM and sanitizes
those TXT files.

```bash
START_DATE=2025-09-01 END_DATE=2025-09-30 ./scripts/run_sep_2025_ctm_ptm_overnight.sh
```

Useful worker settings for this laptop:

```bash
START_DATE=2025-09-01 \
END_DATE=2025-09-30 \
CTM_HOUR_WORKERS=16 \
PTM_HOUR_WORKERS=12 \
DOWNLOAD_WORKERS=8 \
./scripts/run_sep_2025_ctm_ptm_overnight.sh
```

The script name still says `sep` because it started as the September wrapper,
but the dates come from `START_DATE` and `END_DATE`.

Outputs land here:

```text
outputs/ctm_hourly_p25/YYYY/MM/DD/
outputs/ptm_hourly_p25/YYYY/MM/DD/
```

Logs land here:

```text
outputs/logs/
```

## Generic Downloader

Use this when you want one product at a time or a product other than CTM/PTM.

```bash
python scripts/process_product_oct_nov_p25.py \
  --product CTM \
  --layer brightness \
  --start-date 2025-10-01 \
  --end-date 2025-10-31 \
  --output-root outputs/ctm_hourly_p25 \
  --hour-workers 16 \
  --download-workers 8
```

PTM example:

```bash
python scripts/process_product_oct_nov_p25.py \
  --product PTM \
  --layer Polar_pB \
  --start-date 2025-10-01 \
  --end-date 2025-10-31 \
  --output-root outputs/ptm_hourly_p25 \
  --hour-workers 12 \
  --download-workers 8
```

The downloader expects NASA folders in this shape:

```text
https://umbra.nascom.nasa.gov/punch/3/PRODUCT/YYYY/MM/DD/
```

If a day is missing from NASA, the log prints `[missing-day]` or `[day-skip]`
and keeps going.

## Local FITS Pile

Use this if the FITS files are already downloaded.

```bash
python scripts/process_local_by_hour_p25.py \
  --product CTM \
  --layer brightness \
  --input-root . \
  --output-root outputs/ctm_local_hourly_p25 \
  --hour-workers 8
```

For nested folders:

```bash
python scripts/process_local_by_hour_p25.py \
  --product PTM \
  --layer Polar_pB \
  --input-root /path/to/fits \
  --recursive \
  --output-root outputs/ptm_local_hourly_p25 \
  --hour-workers 8
```

For one test hour:

```bash
python scripts/process_local_by_hour_p25.py \
  --product CTM \
  --layer brightness \
  --input-root . \
  --hour-filter '^2025090100$' \
  --max-hours 1 \
  --output-root outputs/test_ctm_local
```

## Single File Check

Use this to inspect one FITS file without running a whole month.

```bash
python scripts/run_median_filter.py \
  --input /path/to/PUNCH_L3_CTM_YYYYMMDDHHMMSS_v0k.fits \
  --product CTM \
  --layer brightness \
  --output-dir outputs/median \
  --convert-to-s10
```

## PIM pB

For PIM, `Polar_pB` is computed from three polarizer planes after each plane is
median-binned:

```text
Q = (4 / 3) Z - (2 / 3) (P + M)
U = (2 / sqrt(3)) P - (2 / sqrt(3)) M
pB = sqrt(Q^2 + U^2)
```

## TXT Contract

The Fortran reader is fixed-width, so the writer is strict:

- line type is `L3`
- RA is written as `F6.2`
- DEC is written as `F6.2`
- brightness is written as `F8.2`
- timestamp is written after brightness
- rows with NaN or Inf are skipped
- rows that print as `0.00` are skipped
- brightness outside the `F8.2` range is skipped

Run the sanitizer when you want to re-check an output folder:

```bash
python scripts/sanitize_tomography_txt_fixed_width.py outputs/ctm_hourly_p25 --min-age-seconds 0
```

Dry run:

```bash
python scripts/sanitize_tomography_txt_fixed_width.py outputs/ctm_hourly_p25 --dry-run --min-age-seconds 0
```

## Plotting

```bash
python plot_punch.py outputs/ctm_hourly_p25/2025/10/01/PUNCH_L3_CTM_2025100100_brightness_p25_COMPOSITE.txt
```

The plotter uses a fixed color range of `-100` to `500` S10 so different hours
can be compared by eye.

## Linux Cluster Note

The worker flags use thread pools inside one Python process. They do not spread
one run across several cluster nodes.

On a cluster, split the date range yourself through the scheduler. For example,
run one job per week or one job per day, each with a different `--start-date`
and `--end-date`.

## Where The Documentation Lives

The Obsidian vault is in:

```text
punch_pipeline_obsidian_vault/
```

Start with `00 Start Here.md`.

## Git Hygiene

Generated data should stay out of git:

```text
outputs/
*.fits
*.part
```

Move generated TXT products to another machine with `rsync` or `scp`.
