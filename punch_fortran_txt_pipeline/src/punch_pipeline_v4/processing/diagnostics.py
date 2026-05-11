from __future__ import annotations

import numpy as np


def array_stats(values, name: str = "values") -> dict:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    zeros = finite & (arr == 0.0)
    negative = finite & (arr < 0.0)
    stats = {
        f"{name}_shape": tuple(arr.shape),
        f"{name}_finite_count": int(np.sum(finite)),
        f"{name}_nan_count": int(np.sum(np.isnan(arr))),
        f"{name}_zero_count": int(np.sum(zeros)),
        f"{name}_negative_count": int(np.sum(negative)),
    }
    if np.any(finite):
        vals = arr[finite]
        stats.update({
            f"{name}_min": float(np.min(vals)),
            f"{name}_median": float(np.median(vals)),
            f"{name}_max": float(np.max(vals)),
        })
    else:
        stats.update({f"{name}_min": None, f"{name}_median": None, f"{name}_max": None})
    return stats
