from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.time import Time
from ..models import BinnedMap


def read_tomography_txt(path: str | Path, product: str = "UNKNOWN", layer_name: str = "txt") -> dict:
    """Read existing tomography TXT into flat arrays.

    This intentionally returns a simple dict because reconstructing a 2D grid from
    arbitrary TXT rows may be product-specific. Composite-from-TXT can still use
    these arrays after later gridding logic is added.
    """
    path = Path(path).expanduser()
    rows = []
    with path.open() as f:
        header = f.readline().strip()
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3]), parts[4]))
    if rows:
        level, coord1, coord2, brightness, times = zip(*rows)
    else:
        level, coord1, coord2, brightness, times = [], [], [], [], []
    return {
        "path": str(path),
        "header": header,
        "level": np.array(level),
        "coord1": np.array(coord1, dtype=float),
        "coord2": np.array(coord2, dtype=float),
        "brightness": np.array(brightness, dtype=float),
        "time": np.array(times),
        "product": product,
        "layer_name": layer_name,
    }
