from __future__ import annotations

import numpy as np

S10_COEFF = 4.5e-16
S10_MIN_DEFAULT = -500.0
S10_MAX_DEFAULT = 2000.0


def native_to_s10(values, coeff: float = S10_COEFF):
    """Convert native PUNCH pB-like brightness units to S10."""
    return np.asarray(values, dtype=np.float64) / coeff


def s10_valid_mask(
    values_s10,
    *,
    s10_min: float = S10_MIN_DEFAULT,
    s10_max: float = S10_MAX_DEFAULT,
    drop_zero: bool = True,
    positive_only: bool = False,
):
    values_s10 = np.asarray(values_s10, dtype=np.float64)
    mask = np.isfinite(values_s10) & (values_s10 >= s10_min) & (values_s10 < s10_max)
    if drop_zero:
        mask &= values_s10 != 0.0
    if positive_only:
        mask &= values_s10 > 0.0
    return mask


def apply_s10_filter(values_s10, **kwargs):
    out = np.full(np.asarray(values_s10).shape, np.nan, dtype=np.float64)
    mask = s10_valid_mask(values_s10, **kwargs)
    out[mask] = np.asarray(values_s10, dtype=np.float64)[mask]
    return out, mask
