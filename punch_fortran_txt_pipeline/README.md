# PUNCH Fortran TXT Pipeline

Clean, GitHub-ready scaffold for producing fixed-width tomography TXT files from
PUNCH L3 FITS products.

The main production workflow downloads PUNCH L3 FITS files from NASA, bins each
image to a 1 degree sky grid, computes hourly p25 composites with zero bins
treated as missing, and writes one Fortran-readable TXT file per hour.

## Supported Products

| Product | Status | Notes |
| --- | --- | --- |
| CTM | Production-used | Full Oct/Nov hourly p25 downloader/compositor included. |
| PTM | Production-used | Full Oct/Nov hourly p25 downloader/compositor included for `Polar_pB`. |
| CIM | Smoke-tested | Single-image and local hourly tools work for 2D `brightness` maps. |
| PIM | Smoke-tested | `Polar_pB` is computed from `Polar_M`, `Polar_Z`, `Polar_P` by median-binning M/Z/P separately, then computing pB. |
| PAM | Smoke-tested | Same 3-layer style as PTM; `Polar_pB` single-image binning passed against a NASA sample. |
| CAM | Not included | Adapter was still a placeholder, so it is intentionally omitted. |

## Modular Architecture

The pipeline is built around product adapters:

```text
adapter -> LayerFrame or BinnedMap -> hourly p25 composite -> fixed-width TXT
```

To add a new product:

1. Add an adapter in `src/punch_pipeline_v4/adapters/`.
2. Register it in `src/punch_pipeline_v4/adapters/registry.py`.
3. Run the generic scripts with `--product NEWPRODUCT --layer LAYERNAME`.

Simple image products only need `load_layer()`. Products with special science
logic can implement `make_binned_map()`, like PIM does for `Polar_pB`.

## Install

Use a Python environment with the scientific stack installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If you are using an existing Anaconda environment with `numpy`, `astropy`,
`scipy`, `pandas`, and `matplotlib`, you can run the scripts directly.

## Full CTM Run

```bash
./scripts/run_ctm_oct_nov_overnight.sh
```

Equivalent generic command:

```bash
python scripts/process_product_oct_nov_p25.py \
  --product CTM \
  --layer brightness \
  --start-date 2025-10-01 \
  --end-date 2025-11-30 \
  --output-root outputs/ctm_hourly_p25 \
  --hour-workers 4 \
  --download-workers 8
```

Outputs:

```text
outputs/ctm_hourly_p25/YYYY/MM/DD/PUNCH_L3_CTM_YYYYMMDDHH_brightness_p25_COMPOSITE.txt
```

## Full PTM Run

```bash
./scripts/run_ptm_oct_nov_overnight.sh
```

Equivalent generic command:

```bash
python scripts/process_product_oct_nov_p25.py \
  --product PTM \
  --layer Polar_pB \
  --start-date 2025-10-01 \
  --end-date 2025-11-30 \
  --output-root outputs/ptm_hourly_p25 \
  --hour-workers 6 \
  --download-workers 8
```

Outputs:

```text
outputs/ptm_hourly_p25/YYYY/MM/DD/PUNCH_L3_PTM_YYYYMMDDHH_Polar_pB_p25_COMPOSITE.txt
```

## Generic NASA Downloader

The generic downloader works for registered products whose NASA directory follows:

```text
https://umbra.nascom.nasa.gov/punch/3/PRODUCT/YYYY/MM/DD/
```

Examples:

```bash
python scripts/process_product_oct_nov_p25.py \
  --product CIM \
  --layer brightness \
  --start-date 2025-10-01 \
  --end-date 2025-10-02 \
  --output-root outputs/cim_hourly_p25
```

```bash
python scripts/process_product_oct_nov_p25.py \
  --product PIM \
  --layer Polar_pB \
  --start-date 2025-10-01 \
  --end-date 2025-10-02 \
  --output-root outputs/pim_hourly_p25
```

```bash
python scripts/process_product_oct_nov_p25.py \
  --product PAM \
  --layer Polar_pB \
  --start-date 2025-10-01 \
  --end-date 2025-10-02 \
  --output-root outputs/pam_hourly_p25
```

Use `--base-url` if a product lives somewhere else.

### Linux Cluster Note

This code is compatible with Linux as normal Python, assuming the environment has
the dependencies installed and the machine can reach the NASA URLs.

The `--hour-workers` and `--download-workers` flags are **single-process,
single-node thread pools**. They do not automatically use multiple cluster nodes.
On a multi-node Fortran machine or HPC cluster, run this on one compute node, or
split work explicitly with the scheduler by date ranges, for example:

```bash
python scripts/process_product_oct_nov_p25.py --product CTM --start-date 2025-10-01 --end-date 2025-10-07
python scripts/process_product_oct_nov_p25.py --product CTM --start-date 2025-10-08 --end-date 2025-10-14
```

For SLURM/PBS-style multi-node processing, the right pattern is usually a job
array where each job gets a separate date range or day. The current scripts will
not distribute one run across nodes by themselves.

## Single FITS Median Binning

CIM example:

```bash
python scripts/run_median_filter.py \
  --input /path/to/PUNCH_L3_CIM_YYYYMMDDHHMMSS_v0k.fits \
  --product CIM \
  --layer brightness \
  --output-dir outputs/cim_median \
  --bin-size-deg 1.0 \
  --convert-to-s10
```

PIM `Polar_pB` example:

```bash
python scripts/run_median_filter.py \
  --input /path/to/PUNCH_L3_PIM_YYYYMMDDHHMMSS_v0k.fits \
  --product PIM \
  --layer Polar_pB \
  --output-dir outputs/pim_median \
  --bin-size-deg 1.0 \
  --convert-to-s10
```

For PIM, `Polar_pB` is not read as one raw plane. The adapter median-bins the
three polarizer planes first, then computes:

```text
Q = (4/3) Z - (2/3) (P + M)
U = (2/sqrt(3)) P - (2/sqrt(3)) M
pB = sqrt(Q^2 + U^2)
```

## Local Hourly Composite From Existing FITS

If you already downloaded a lot of FITS files, use the local by-hour script. It
scans a folder, groups files by `YYYYMMDDHH` from the filename, and writes one
p25 TXT per hour.

If the FITS files are in the pipeline root:

```bash
python scripts/process_local_by_hour_p25.py \
  --product CTM \
  --layer brightness \
  --input-root . \
  --output-root outputs/ctm_local_hourly_p25 \
  --hour-workers 4
```

Recursive search:

```bash
python scripts/process_local_by_hour_p25.py \
  --product PTM \
  --layer Polar_pB \
  --input-root /path/to/downloaded/fits \
  --recursive \
  --output-root outputs/ptm_local_hourly_p25 \
  --hour-workers 4
```

Small test on one hour:

```bash
python scripts/process_local_by_hour_p25.py \
  --product PIM \
  --layer Polar_pB \
  --input-root . \
  --hour-filter '^2025090100$' \
  --max-hours 1 \
  --output-root outputs/test_pim_local
```

The older `run_hourly_composite.py` script is still useful when you want to
manually provide exactly one hour of files:

```bash
python scripts/run_hourly_composite.py \
  --input-glob '/path/to/hour/*.fits' \
  --product PIM \
  --layer Polar_pB \
  --output-dir outputs/pim_hourly \
  --composite-method percentile \
  --percentile 25 \
  --convert-to-s10
```

## Fortran TXT Safety Rules

The writer protects the fixed-width Fortran reader:

- rows are written as `L3`, RA `F6.2`, DEC `F6.2`, brightness `F8.2`, then timestamp
- zero values are omitted
- values that would print as `0.00` are omitted
- brightness values outside the `F8.2` range are omitted
- NaN/Inf rows are omitted

To re-check or clean an output folder:

```bash
python scripts/sanitize_tomography_txt_fixed_width.py outputs/ctm_hourly_p25 --min-age-seconds 0
python scripts/sanitize_tomography_txt_fixed_width.py outputs/ptm_hourly_p25 --min-age-seconds 0
```

Dry run:

```bash
python scripts/sanitize_tomography_txt_fixed_width.py outputs/ctm_hourly_p25 --dry-run --min-age-seconds 0
```

## Plotting

```bash
python plot_punch.py outputs/ctm_hourly_p25/2025/10/01/PUNCH_L3_CTM_2025100100_brightness_p25_COMPOSITE.txt
```

The plotter uses the fixed display range `-100..500` S10 for visual comparison.

## Repository Hygiene

Generated products are intentionally ignored by git:

```text
outputs/
*.fits
*.part
```

That keeps this repo small enough to push to GitHub. Transfer generated TXT
products separately with `rsync` when needed.
