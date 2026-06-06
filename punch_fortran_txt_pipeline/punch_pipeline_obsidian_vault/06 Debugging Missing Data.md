# Debugging Missing Data

Missing NASA data is normal for some products and dates.

The downloader logs these cases instead of stopping:

```text
[missing-day] URL
[day-skip] YYYY-MM-DD: no PRODUCT files found
```

That usually means the directory is absent or empty on the NASA site.

## Check A Run

Look at the final line:

```text
[done] days=... hours_seen=... hours_done=... hours_skipped=... files_downloaded=...
```

Then compare the output folder with the date range.

Useful checks:

```bash
find outputs/ctm_hourly_p25/2025/10 -name '*.txt' | wc -l
find outputs/ptm_hourly_p25/2025/10 -name '*.txt' | wc -l
```

For a full month with all hours present, expect:

```text
30 day month: 720 files
31 day month: 744 files
```

Fewer files can be fine if NASA did not publish that product for some days.

## Runtime Warning: All-NaN Slice

This warning usually means a sky bin had no valid samples after zero removal.
It does not mean the whole FITS file is bad.

The writer skips bins that remain NaN.
