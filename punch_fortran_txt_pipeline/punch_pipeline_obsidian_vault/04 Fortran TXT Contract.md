# Fortran TXT Contract

The tomography reader expects fixed-width rows. Do not casually change the row
format.

Each data row looks like:

```text
L3    RA    DEC BRIGHT  timestamp
```

The actual writer uses:

```text
L3  {ra:F6.2} {dec:F6.2}{brightness:F8.2} {timestamp}
```

The fields are:

- `L3`
- RA as `F6.2`
- DEC as `F6.2`
- brightness as `F8.2`
- ISO timestamp

## Rows The Writer Drops

The writer skips rows with:

- NaN or Inf coordinates
- NaN or Inf brightness
- empty timestamp
- brightness that prints as `0.00`
- brightness outside the `F8.2` range

The `F8.2` brightness range is:

```text
-9999.99 to 99999.99
```

## Sanitizer

Run this after a batch if you want a second pass over the TXT files:

```bash
python scripts/sanitize_tomography_txt_fixed_width.py outputs/ctm_hourly_p25 --min-age-seconds 0
```

Use `--dry-run` first if you want counts without edits.
