# Commands

## Monthly CTM And PTM

```bash
START_DATE=2025-09-01 \
END_DATE=2025-09-30 \
CTM_HOUR_WORKERS=16 \
PTM_HOUR_WORKERS=12 \
DOWNLOAD_WORKERS=8 \
./scripts/run_sep_2025_ctm_ptm_overnight.sh
```

The wrapper runs CTM first, sanitizes CTM TXT files, runs PTM, then sanitizes
PTM TXT files.

## CTM Only

```bash
python scripts/process_product_oct_nov_p25.py \
  --product CTM \
  --layer brightness \
  --start-date 2025-09-01 \
  --end-date 2025-09-30 \
  --output-root outputs/ctm_hourly_p25 \
  --hour-workers 16 \
  --download-workers 8
```

## PTM Only

```bash
python scripts/process_product_oct_nov_p25.py \
  --product PTM \
  --layer Polar_pB \
  --start-date 2025-09-01 \
  --end-date 2025-09-30 \
  --output-root outputs/ptm_hourly_p25 \
  --hour-workers 12 \
  --download-workers 8
```

## Local FITS Files

```bash
python scripts/process_local_by_hour_p25.py \
  --product CTM \
  --layer brightness \
  --input-root /path/to/fits \
  --recursive \
  --output-root outputs/ctm_local_hourly_p25 \
  --hour-workers 8
```

## Rebuild Existing Outputs

Add `--overwrite` to the processing command.

Use this after changing the compositing logic. Existing TXT files do not update
themselves.
