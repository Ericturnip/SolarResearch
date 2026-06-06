from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.time import Time
from ..models import BinnedMap
from ..processing.units import s10_valid_mask


def year_doy_fraction_string(t_ref: Time) -> str:
    dt = t_ref.to_datetime()
    jan1 = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    doy_fraction = t_ref.jd - Time(jan1, scale="utc").jd + 1.0
    return f"{dt.year} {doy_fraction:.8f}"


def _coords_for_ascii(bmap: BinnedMap):
    hpln = bmap.hpln_centers.ravel()
    hplt = bmap.hplt_centers.ravel()
    if bmap.solar_wcs is not None and bmap.radec_wcs is not None:
        px, py = bmap.solar_wcs.world_to_pixel_values(hpln, hplt)
        ra, dec = bmap.radec_wcs.pixel_to_world_values(px, py)
        return np.asarray(ra), np.asarray(dec)
    return hpln, hplt


def write_tomography_txt(
    bmap: BinnedMap,
    output_path: str | Path,
    *,
    values_are_s10: bool = True,
    omit_empty_bins: bool = True,
    drop_zero_bins: bool = True,
    positive_only: bool = False,
    s10_min: float = -500.0,
    s10_max: float = 2000.0,
    enforce_fixed_width: bool = True,
) -> dict:
    """Write tomography-style TXT from a binned map.

    Assumes `bmap.values` are already in the desired output unit, normally S10.
    """
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(bmap.values, dtype=np.float64).ravel()
    ra, dec = _coords_for_ascii(bmap)
    if bmap.time_map is None:
        times = np.full(values.shape, bmap.timestamp.to_datetime().isoformat(), dtype="<U30")
    else:
        times = bmap.time_map.ravel()

    valid = np.isfinite(values) & np.isfinite(ra) & np.isfinite(dec) & (times != "")
    if values_are_s10:
        valid &= s10_valid_mask(values, s10_min=s10_min, s10_max=s10_max, drop_zero=drop_zero_bins, positive_only=positive_only)
    elif omit_empty_bins:
        valid &= np.isfinite(values)
        if drop_zero_bins:
            valid &= np.round(values, 2) != 0.0
    pre_width_valid = valid.copy()
    if enforce_fixed_width:
        # The Fortran reader expects brightness in an F8.2 field.
        valid &= (values >= -9999.99) & (values <= 99999.99)

    with output_path.open("w") as f:
        f.write(f"{year_doy_fraction_string(bmap.timestamp)}\n")
        for r, d, b, tm in zip(ra[valid], dec[valid], values[valid], times[valid]):
            f.write(f"L3  {r:6.2f} {d:6.2f}{b:8.2f} {tm}\n")

    return {
        "output_path": str(output_path),
        "rows_written": int(np.sum(valid)),
        "candidate_rows": int(values.size),
        "removed_rows": int(values.size - np.sum(valid)),
        "removed_fixed_width_overflow": int(np.sum(pre_width_valid) - np.sum(valid)),
        "unique_timestamps_written": int(len(set(times[valid]))),
    }
