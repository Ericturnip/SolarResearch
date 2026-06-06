from __future__ import annotations

import numpy as np
from astropy.time import Time
from ..models import BinnedMap, CompositeResult
from .diagnostics import array_stats


def _statistic(stack: np.ndarray, method: str, percentile: float | None = None):
    method = method.lower()
    if method == "nanmedian":
        return np.nanmedian(stack, axis=0)
    if method == "nanmean":
        return np.nanmean(stack, axis=0)
    if method == "nanmin":
        return np.nanmin(stack, axis=0)
    if method in {"p30", "percentile"}:
        p = 30.0 if percentile is None else percentile
        return np.nanpercentile(stack, p, axis=0)
    raise ValueError(f"Unknown composite method: {method}")


def _closest_source_values_and_time_map(stack, target, input_time_maps):
    """Choose the real input sample nearest to the composite target for each bin."""
    diffs = np.abs(stack - target[None, :, :])
    diffs[~np.isfinite(diffs)] = np.inf
    good = np.any(np.isfinite(stack), axis=0) & np.isfinite(target)
    best_idx = np.full(target.shape, -1, dtype=int)
    if np.any(good):
        best_idx[good] = np.argmin(diffs[:, good], axis=0)
    values = np.full(target.shape, np.nan, dtype=np.float64)
    time_map = np.full(target.shape, "", dtype="<U30")
    for i, tm in enumerate(input_time_maps):
        source = tm if tm is not None else np.full(target.shape, "", dtype="<U30")
        m = best_idx == i
        values[m] = stack[i][m]
        time_map[m] = source[m]
    return values, time_map


def composite_binned_maps(
    maps: list[BinnedMap],
    *,
    method: str = "nanmedian",
    percentile: float | None = None,
    drop_zero_before_stat: bool = True,
) -> CompositeResult:
    """Composite a list of same-grid binned maps.

    Zeros are treated as missing by default because minimum/p30 maps should not
    accidentally select artificial zero-fill values.
    """
    if not maps:
        raise ValueError("No maps supplied to composite_binned_maps.")
    ref_shape = maps[0].values.shape
    for m in maps:
        if m.values.shape != ref_shape:
            raise ValueError("All binned maps must have the same shape/grid for compositing.")

    stack = np.stack([np.asarray(m.values, dtype=np.float64) for m in maps], axis=0)
    before_diag = array_stats(stack, "stack_before_zero_filter")
    removed_zero_count = 0
    if drop_zero_before_stat:
        zero_mask = np.isfinite(stack) & (stack == 0.0)
        removed_zero_count = int(np.sum(zero_mask))
        stack = stack.copy()
        stack[zero_mask] = np.nan

    with np.errstate(all="ignore"):
        comp = _statistic(stack, method, percentile=percentile)

    percentile_method = method.lower() in {"p30", "percentile"}
    if percentile_method:
        values, time_map = _closest_source_values_and_time_map(
            stack,
            comp,
            [m.time_map for m in maps],
        )
    else:
        values = comp
        _, time_map = _closest_source_values_and_time_map(
            stack,
            comp,
            [m.time_map for m in maps],
        )

    out = BinnedMap(
        product=maps[0].product,
        layer_name=maps[0].layer_name,
        values=values,
        hpln_centers=maps[0].hpln_centers,
        hplt_centers=maps[0].hplt_centers,
        timestamp=maps[0].timestamp,
        solar_wcs=maps[0].solar_wcs,
        radec_wcs=maps[0].radec_wcs,
        time_map=time_map,
        unit=maps[0].unit,
        metadata={
            "composite_method": method,
            "output_value": "nearest_real_sample_to_percentile" if percentile_method else "computed_statistic",
        },
    )
    unique_times = sorted({str(t) for m in maps for t in ([m.timestamp.isot] if hasattr(m.timestamp, "isot") else [])})
    diag = before_diag | array_stats(stack, "stack_after_zero_filter") | array_stats(comp, "composite")
    diag["removed_zero_count_before_stat"] = removed_zero_count
    diag["unique_timestamps_written"] = int(len(set(time_map.ravel()) - {""}))
    return CompositeResult(out, method=method, input_count=len(maps), unique_input_times=unique_times, diagnostics=diag)
