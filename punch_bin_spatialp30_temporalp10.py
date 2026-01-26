#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
import argparse, glob, os, sys, warnings

warnings.filterwarnings("ignore")

S10_COEFF = 4.5e-16

def _get_date_obs_any_hdu(hdul):
    d = hdul[0].header.get("DATE-OBS") if len(hdul) > 0 else None
    if (not d) and (len(hdul) > 1):
        d = hdul[1].header.get("DATE-OBS")
    return d

def get_timestamp_from_header(header):
    d = header.get("DATE-OBS")
    if d:
        return Time(d, format="isot", scale="utc").strftime("%Y%m%d%H%M%S")
    return "00000000000000"

def load_fits_data(fp):
    try:
        with fits.open(fp) as hdul:
            if hdul[0].data is not None:
                data = hdul[0].data
                hdr = hdul[0].header
            elif len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
                hdr = hdul[1].header
            else:
                return None, None, None, None, None

            data = np.asarray(data).squeeze().astype(np.float64)

            date_obs = _get_date_obs_any_hdu(hdul) or hdr.get("DATE-OBS")
            if not date_obs:
                return None, None, None, None, None
            t = Time(date_obs, format="isot", scale="utc")

            wcs_solar = WCS(hdr)
            try:
                wcs_radec = WCS(hdr, key="A")
            except Exception:
                wcs_radec = None

            return data, t, wcs_solar, wcs_radec, hdr
    except Exception:
        return None, None, None, None, None

def parse_hhmm(s):
    hh, mm = s.split(":")
    return int(hh), int(mm)

def filter_files_by_time(files, t0, t1):
    out = []
    for fp in files:
        try:
            with fits.open(fp) as hdul:
                d = _get_date_obs_any_hdu(hdul)
                if not d:
                    continue
                t = Time(d, format="isot", scale="utc")
                if (t >= t0) and (t < t1):
                    out.append(fp)
        except Exception:
            continue
    return sorted(out)

def process(files, bin_size_deg=1.0, spatial_percentile=30.0, temporal_percentile=10.0):
    if not files:
        print("No files.")
        return

    # init
    data0, t0, wcs_solar0, wcs_radec0, hdr0 = load_fits_data(files[0])
    if data0 is None or wcs_solar0 is None or wcs_radec0 is None:
        print("Init failed (data/WCS missing).")
        return

    H, W = data0.shape
    y_idx, x_idx = np.indices((H, W))

    hpln0, hplt0 = wcs_solar0.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())
    x_bins = np.arange(np.floor(np.nanmin(hpln0)), np.ceil(np.nanmax(hpln0)) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(np.nanmin(hplt0)), np.ceil(np.nanmax(hplt0)) + bin_size_deg, bin_size_deg)

    nbx = len(x_bins) - 1
    nby = len(y_bins) - 1

    # bin centers
    bin_hpln_centers = binned_statistic_2d(hpln0, hplt0, hpln0, statistic="mean", bins=[x_bins, y_bins]).statistic.T
    bin_hplt_centers = binned_statistic_2d(hpln0, hplt0, hplt0, statistic="mean", bins=[x_bins, y_bins]).statistic.T

    start_ts = get_timestamp_from_header(hdr0)
    last_hdr = hdr0

    per_frame = []
    per_time = []

    for i, fp in enumerate(files):
        print(f"[{i+1}/{len(files)}] {os.path.basename(fp)}")
        data, tt, wcs_solar, _, hdr = load_fits_data(fp)
        if data is None or wcs_solar is None:
            continue
        last_hdr = hdr

        flat = (data.flatten() / S10_COEFF).astype(np.float64)
        hpln, hplt = wcs_solar.pixel_to_world_values(x_idx.flatten(), y_idx.flatten())

        xi = np.digitize(hpln, x_bins) - 1
        yi = np.digitize(hplt, y_bins) - 1

        ok = (
            (xi >= 0) & (xi < nbx) &
            (yi >= 0) & (yi < nby) &
            np.isfinite(flat) &
            (flat > 0) & (flat < 2000)
        )

        # spatial percentile per bin for THIS frame
        grid = binned_statistic_2d(
            hpln[ok], hplt[ok], flat[ok],
            statistic=lambda v: np.percentile(v, spatial_percentile),
            bins=[x_bins, y_bins]
        ).statistic.T

        grid[np.isnan(grid)] = np.nan
        per_frame.append(grid)
        per_time.append(tt)

    end_ts = get_timestamp_from_header(last_hdr)

    if not per_frame:
        print("No usable frames processed.")
        return

    stack = np.stack(per_frame, axis=0)  # (nframe, y, x)

    # temporal low-percentile (background) across frames
    bg = np.nanpercentile(stack, temporal_percentile, axis=0)

    # representative time: closest frame to bg per bin
    diffs = np.abs(stack - bg[None, :, :])
    diffs[np.isnan(diffs)] = np.inf
    best_idx = np.argmin(diffs, axis=0)

    time_grid = np.full(bg.shape, "", dtype="<U30")
    for fi, tt in enumerate(per_time):
        m = best_idx == fi
        time_grid[m] = tt.to_datetime().isoformat()

    # convert to RA/Dec
    res_s10 = bg.flatten()
    res_hpln = bin_hpln_centers.flatten()
    res_hplt = bin_hplt_centers.flatten()
    res_time = time_grid.flatten()

    px, py = wcs_solar0.world_to_pixel_values(res_hpln, res_hplt)
    ra, dec = wcs_radec0.pixel_to_world_values(px, py)

    m = np.isfinite(res_s10) & (res_time != "") & np.isfinite(ra) & np.isfinite(dec)

    clean_ra = ra[m]
    clean_dec = dec[m]
    clean_s10 = res_s10[m]
    clean_time = res_time[m]

    # header line
    t_dt = t0.to_datetime()
    jan1 = t_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    jan1_jd = Time(jan1, scale="utc").jd
    doy_fraction = t0.jd - jan1_jd + 1.0
    header_line = f"{t_dt.year} {doy_fraction:.8f}"

    out = f"PUNCH_L3_CIM_RANGE_{start_ts}_{end_ts}_sp{int(spatial_percentile)}_tp{int(temporal_percentile)}_bin.txt"
    print(f"Writing {len(clean_s10)} points -> {out}")

    with open(out, "w") as f:
        f.write(header_line + "\n")
        for r, d, b, tm in zip(clean_ra, clean_dec, clean_s10, clean_time):
            f.write(f"L3  {r:6.2f} {d:6.2f}  {b:6.2f} {tm}\n")

    print("Done.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--pattern", default="*_v0i.fits")
    ap.add_argument("--date", required=True)
    ap.add_argument("--start", default="00:00")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--bin_size_deg", type=float, default=1.0)
    ap.add_argument("--spatial_p", type=float, default=30.0, help="spatial percentile inside each bin per frame")
    ap.add_argument("--temporal_p", type=float, default=10.0, help="temporal percentile across frames per bin")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        print(f"No files (dir={args.dir}, pattern={args.pattern})")
        sys.exit(1)

    hh, mm = parse_hhmm(args.start)
    t0 = Time(f"{args.date}T{hh:02d}:{mm:02d}:00", format="isot", scale="utc")
    t1 = t0 + TimeDelta(args.hours * 3600.0, format="sec")

    print(f"Window: [{t0.isot}, {t1.isot})")
    files = filter_files_by_time(files, t0, t1)
    print("Selected", len(files), "files")
    if not files:
        sys.exit(1)

    process(files, bin_size_deg=args.bin_size_deg, spatial_percentile=args.spatial_p, temporal_percentile=args.temporal_p)

if __name__ == "__main__":
    main()
