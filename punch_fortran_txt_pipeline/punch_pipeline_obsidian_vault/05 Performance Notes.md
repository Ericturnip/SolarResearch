# Performance Notes

The laptop has enough RAM to run several hours at once. The limiting factor is
usually network behavior from the NASA server, not local memory.

Good starting settings:

```bash
CTM_HOUR_WORKERS=16
PTM_HOUR_WORKERS=12
DOWNLOAD_WORKERS=8
```

Raising `DOWNLOAD_WORKERS` too far can make downloads retry more often. If the
speed stops improving, the server is probably the bottleneck.

## What The Workers Mean

`--hour-workers` controls how many hours run at once.

`--download-workers` controls how many FITS downloads run inside each hour.

These are Python threads inside one process. They do not use several cluster
nodes by themselves.

## Cluster Use

On SLURM or PBS, split by date range. One job per day or week is easier to
monitor than one huge cross-node job.

Example:

```bash
python scripts/process_product_oct_nov_p25.py --product CTM --start-date 2025-10-01 --end-date 2025-10-07
python scripts/process_product_oct_nov_p25.py --product CTM --start-date 2025-10-08 --end-date 2025-10-14
```
