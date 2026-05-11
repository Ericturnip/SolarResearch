from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np
from astropy.time import Time


@dataclass
class LayerFrame:
    """One product frame normalized by an adapter.

    Product adapters should return this object regardless of whether the source is
    CIM/CTM/PIM/PTM/PAM/CAM/etc.
    """

    product: str
    layer_name: str
    data: np.ndarray
    timestamp: Time
    solar_wcs: Any
    radec_wcs: Any | None = None
    native_unit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class BinnedMap:
    """A 2D sky-binned map plus per-bin coordinates and optional source times."""

    product: str
    layer_name: str
    values: np.ndarray
    hpln_centers: np.ndarray
    hplt_centers: np.ndarray
    timestamp: Time
    solar_wcs: Any | None = None
    radec_wcs: Any | None = None
    time_map: np.ndarray | None = None
    unit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CompositeResult:
    """Output from temporal compositing of binned maps."""

    binned_map: BinnedMap
    method: str
    input_count: int
    unique_input_times: list[str]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
