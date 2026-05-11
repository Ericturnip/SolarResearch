from __future__ import annotations

import numpy as np
from scipy.stats import binned_statistic_2d
from .diagnostics import array_stats
from ..models import LayerFrame, BinnedMap
from .units import S10_COEFF, native_to_s10


def build_grid_from_wcs(solar_wcs, shape, bin_size_deg: float = 1.0):
    """Build bin edges and bin-center coordinate arrays from a 2D WCS/image shape."""
    h, w = shape
    y_idx, x_idx = np.indices((h, w))
    hpln, hplt = solar_wcs.pixel_to_world_values(x_idx.ravel(), y_idx.ravel())
    finite_xy = np.isfinite(hpln) & np.isfinite(hplt)
    if not np.any(finite_xy):
        raise ValueError("No finite WCS coordinates found while building grid.")

    hpln_f = hpln[finite_xy]
    hplt_f = hplt[finite_xy]
    x_bins = np.arange(np.floor(np.nanmin(hpln_f)), np.ceil(np.nanmax(hpln_f)) + bin_size_deg, bin_size_deg)
    y_bins = np.arange(np.floor(np.nanmin(hplt_f)), np.ceil(np.nanmax(hplt_f)) + bin_size_deg, bin_size_deg)

    hpln_centers = binned_statistic_2d(hpln_f, hplt_f, hpln_f, statistic="mean", bins=[x_bins, y_bins]).statistic.T
    hplt_centers = binned_statistic_2d(hpln_f, hplt_f, hplt_f, statistic="mean", bins=[x_bins, y_bins]).statistic.T

    return {
        "x_bins": x_bins,
        "y_bins": y_bins,
        "x_idx": x_idx,
        "y_idx": y_idx,
        "hpln_centers": hpln_centers,
        "hplt_centers": hplt_centers,
    }


def per_bin_median(vx, vy, vv, x_bins, y_bins):
    """Fast per-bin median. Output shape is (ny, nx)."""
    nx = len(x_bins) - 1
    ny = len(y_bins) - 1
    ix = np.digitize(vx, x_bins) - 1
    iy = np.digitize(vy, y_bins) - 1
    good = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(vv)
    ix = ix[good]
    iy = iy[good]
    vv = vv[good]
    out = np.full(ny * nx, np.nan, dtype=np.float64)
    if vv.size == 0:
        return out.reshape((ny, nx))
    bid = iy * nx + ix
    order = np.argsort(bid)
    bid_s = bid[order]
    vv_s = vv[order]
    starts = np.r_[0, np.flatnonzero(np.diff(bid_s)) + 1]
    ends = np.r_[starts[1:], bid_s.size]
    for s, e in zip(starts, ends):
        out[bid_s[s]] = np.median(vv_s[s:e])
    return out.reshape((ny, nx))


def median_bin_layer_frame(
    frame: LayerFrame,
    bin_size_deg: float = 1.0,
    convert_to_s10: bool = False,
) -> tuple[BinnedMap, dict]:
    """Median-filter one image by binning it into sky-coordinate bins."""
    if frame.data.ndim != 2:
        raise ValueError(f"median_bin_layer_frame expects 2D data, got shape {frame.data.shape}")

    grid = build_grid_from_wcs(frame.solar_wcs, frame.data.shape, bin_size_deg=bin_size_deg)
    hpln, hplt = frame.solar_wcs.pixel_to_world_values(grid["x_idx"].ravel(), grid["y_idx"].ravel())

    data = np.asarray(frame.data, dtype=np.float64)

    if convert_to_s10:
        vals = native_to_s10(data).ravel()
        out_unit = "S10"
    else:
        vals = data.ravel()
        out_unit = frame.native_unit

    finite_xy = np.isfinite(hpln) & np.isfinite(hplt)
    keep = finite_xy & np.isfinite(vals)

    binned = per_bin_median(hpln[keep], hplt[keep], vals[keep], grid["x_bins"], grid["y_bins"])

    time_map = np.full(binned.shape, frame.timestamp.to_datetime().isoformat(), dtype="<U30")
    time_map[~np.isfinite(binned)] = ""

    metadata = dict(frame.metadata)
    metadata["converted_to_s10_before_binning"] = bool(convert_to_s10)

    out = BinnedMap(
        product=frame.product,
        layer_name=frame.layer_name,
        values=binned,
        hpln_centers=grid["hpln_centers"],
        hplt_centers=grid["hplt_centers"],
        timestamp=frame.timestamp,
        solar_wcs=frame.solar_wcs,
        radec_wcs=frame.radec_wcs,
        time_map=time_map,
        unit=out_unit,
        metadata=metadata,
    )

    diag = array_stats(frame.data, "input") | array_stats(binned, "binned")
    diag["bin_size_deg"] = bin_size_deg
    diag["converted_to_s10_before_binning"] = bool(convert_to_s10)

    return out, diag
