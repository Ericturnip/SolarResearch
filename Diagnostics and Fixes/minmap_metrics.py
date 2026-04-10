#!/usr/bin/env python3
"""
Compute min-map diagnostics from a PUNCH min-bin txt file.

Expected format:
Line 1: "<year> <doy_fraction>"
Then lines like:
L3  <RA> <Dec>  <S10> <ISO_TIME>

Outputs:
- Background stats: median, p90, p95
- Outlier fractions: >2, >10, >50 S10
- Adjacent-pixel smoothness on a 1° RA/Dec grid:
  median and p90 of |ΔS10| between 4-neighbors
"""

import sys
import numpy as np

def load_txt(path: str):
    ra, dec, s10 = [], [], []
    with open(path, "r") as f:
        header = f.readline().strip()  # year + doy_fraction
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expect: ["L3", RA, Dec, S10, Time]
            if len(parts) < 5 or parts[0] != "L3":
                continue
            try:
                ra.append(float(parts[1]))
                dec.append(float(parts[2]))
                s10.append(float(parts[3]))
            except ValueError:
                continue
    return header, np.array(ra), np.array(dec), np.array(s10)

def gridify_1deg(ra, dec, s10):
    # 1° bins in RA/Dec (simple, no spherical area correction)
    ra0 = np.floor(np.nanmin(ra))
    dec0 = np.floor(np.nanmin(dec))

    ix = np.floor(ra - ra0).astype(int)
    iy = np.floor(dec - dec0).astype(int)

    nx = ix.max() + 1
    ny = iy.max() + 1

    grid = np.full((ny, nx), np.nan, dtype=float)
    grid[iy, ix] = s10  # should be unique per 1° bin in your file

    return grid, ra0, dec0

def neighbor_delta_stats(grid):
    # 4-neighbor absolute deltas, ignoring NaNs
    deltas = []

    # vertical neighbors
    a = grid[:-1, :]
    b = grid[1:, :]
    m = np.isfinite(a) & np.isfinite(b)
    deltas.append(np.abs(a[m] - b[m]))

    # horizontal neighbors
    a = grid[:, :-1]
    b = grid[:, 1:]
    m = np.isfinite(a) & np.isfinite(b)
    deltas.append(np.abs(a[m] - b[m]))

    if not deltas:
        return np.array([])

    return np.concatenate(deltas)

def main(path: str):
    header, ra, dec, s10 = load_txt(path)
    good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(s10)
    ra, dec, s10 = ra[good], dec[good], s10[good]

    # --- Background distribution stats ---
    med = np.median(s10)
    p90 = np.percentile(s10, 90)
    p95 = np.percentile(s10, 95)

    # --- Outliers (tomography killers / unmasked stray light) ---
    frac_gt2  = np.mean(s10 > 2.0)
    frac_gt10 = np.mean(s10 > 10.0)
    frac_gt50 = np.mean(s10 > 50.0)

    # --- Adjacent-pixel smoothness on 1° grid ---
    grid, ra0, dec0 = gridify_1deg(ra, dec, s10)
    deltas = neighbor_delta_stats(grid)
    if deltas.size:
        d_med = np.median(deltas)
        d_p90 = np.percentile(deltas, 90)
        d_p95 = np.percentile(deltas, 95)
        n_pairs = deltas.size
    else:
        d_med = d_p90 = d_p95 = np.nan
        n_pairs = 0

    # --- Print report ---
    print(f"File: {path}")
    print(f"Header: {header}")
    print(f"N points: {len(s10)}")
    print()
    print("Background S10 stats (min-map output):")
    print(f"  median  = {med:.4g}")
    print(f"  p90     = {p90:.4g}")
    print(f"  p95     = {p95:.4g}")
    print()
    print("Outlier fractions:")
    print(f"  frac(S10 > 2)   = {100*frac_gt2:.3f}%")
    print(f"  frac(S10 > 10)  = {100*frac_gt10:.3f}%")
    print(f"  frac(S10 > 50)  = {100*frac_gt50:.3f}%")
    print()
    print("Adjacent-pixel |ΔS10| on 1° RA/Dec grid (4-neighbors):")
    print(f"  neighbor pairs = {n_pairs}")
    print(f"  median |Δ|     = {d_med:.4g}")
    print(f"  p90 |Δ|        = {d_p90:.4g}")
    print(f"  p95 |Δ|        = {d_p95:.4g}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python minmap_metrics.py PUNCH_L3_..._min_bin.txt")
        sys.exit(1)
    main(sys.argv[1])