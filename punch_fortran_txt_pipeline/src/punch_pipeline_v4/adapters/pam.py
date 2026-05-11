from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from .base import ProductAdapter
from ..models import LayerFrame


class PAMAdapter(ProductAdapter):
    """Adapter for PUNCH L3 PAM files.

    PAM is a 3-layer polarized low-noise mosaic datacube:
    - layer 0: Polar_B
    - layer 1: Polar_pB
    - layer 2: Polar_pBp

    Current use case: select Polar_pB and pass it through the shared
    median-binning and hourly-composite pipeline.
    """

    product = "PAM"
    default_layer = "Polar_pB"

    layer_index = {
        "polar_b": 0,
        "b": 0,
        "polar_pb": 1,
        "pb": 1,
        "pB": 1,
        "Polar_pB": 1,
        "polar_pbp": 2,
        "pbp": 2,
        "pBp": 2,
        "Polar_pBp": 2,
    }

    canonical_layer_names = {
        0: "Polar_B",
        1: "Polar_pB",
        2: "Polar_pBp",
    }

    def list_layers(self, path: str | Path) -> list[str]:
        return ["Polar_B", "Polar_pB", "Polar_pBp"]

    def load_layer(self, path: str | Path, layer: str | None = None) -> LayerFrame:
        requested = layer or self.default_layer
        key = requested.strip()
        lookup_key = key if key in self.layer_index else key.lower()

        if lookup_key not in self.layer_index:
            raise ValueError(
                f"Unknown PAM layer {requested!r}. "
                f"Allowed layers: {self.list_layers(path)} "
                f"or aliases: {sorted(self.layer_index)}"
            )

        layer_i = self.layer_index[lookup_key]
        canonical_name = self.canonical_layer_names[layer_i]

        path = Path(path).expanduser()

        with fits.open(path, memmap=False) as hdul:
            hdu, header = self._find_science_cube_hdu(hdul)
            cube = np.asarray(hdu.data, dtype=np.float64).squeeze()

            if cube.ndim != 3:
                raise ValueError(f"PAM science data must be 3D; got shape {cube.shape} from {path}")

            if cube.shape[0] == 3:
                data = cube[layer_i, :, :]
            elif cube.shape[-1] == 3:
                data = cube[:, :, layer_i]
            else:
                raise ValueError(
                    f"Cannot identify PAM layer axis. Expected one axis of length 3; got shape {cube.shape}"
                )

            primary_header = hdul[0].header if len(hdul) > 0 else header
            timestamp = self._timestamp_from_headers(header, primary_header)

            solar_wcs = WCS(header).celestial
            radec_wcs = self._try_radec_wcs(header)

            native_unit = header.get("BUNIT") or primary_header.get("BUNIT")

            metadata: dict[str, Any] = {
                "source_path": str(path),
                "filename": path.name,
                "hdu_index": hdul.index_of(hdu) if hasattr(hdul, "index_of") else None,
                "date_beg": header.get("DATE-BEG") or primary_header.get("DATE-BEG"),
                "date_obs": header.get("DATE-OBS") or primary_header.get("DATE-OBS"),
                "date_avg": header.get("DATE-AVG") or primary_header.get("DATE-AVG"),
                "date_end": header.get("DATE-END") or primary_header.get("DATE-END"),
                "bunit": native_unit,
                "ctype1": header.get("CTYPE1"),
                "ctype2": header.get("CTYPE2"),
                "ctype3": header.get("CTYPE3"),
                "ctype1a": header.get("CTYPE1A"),
                "ctype2a": header.get("CTYPE2A"),
                "ctype3a": header.get("CTYPE3A"),
                "shape": tuple(cube.shape),
                "selected_layer_index": layer_i,
                "selected_layer_name": canonical_name,
                "obslayr1": header.get("OBSLAYR1"),
                "obslayr2": header.get("OBSLAYR2"),
                "obslayr3": header.get("OBSLAYR3"),
                "obs_mode": header.get("OBS-MODE"),
                "level": header.get("LEVEL") or primary_header.get("LEVEL"),
                "typecode": header.get("TYPECODE") or primary_header.get("TYPECODE"),
            }

        return LayerFrame(
            product=self.product,
            layer_name=canonical_name,
            data=data,
            timestamp=timestamp,
            solar_wcs=solar_wcs,
            radec_wcs=radec_wcs,
            native_unit=native_unit,
            metadata=metadata,
        )

    @staticmethod
    def _find_science_cube_hdu(hdul: fits.HDUList):
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue

            arr = np.asarray(data).squeeze()
            if arr.ndim == 3 and 3 in arr.shape:
                return hdu, hdu.header

        shapes = []
        for i, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            shapes.append(
                (
                    i,
                    type(hdu).__name__,
                    None if data is None else getattr(data, "shape", None),
                )
            )

        raise ValueError(f"Could not find a 3D PAM science cube HDU. HDUs seen: {shapes}")

    @staticmethod
    def _timestamp_from_headers(*headers) -> Time:
        for key in ("DATE-AVG", "DATE-OBS", "DATE-BEG", "DATE-END"):
            for header in headers:
                value = header.get(key) if header is not None else None
                if value:
                    return Time(value, format="isot", scale="utc")

        raise ValueError("PAM file has no DATE-AVG/DATE-OBS/DATE-BEG/DATE-END timestamp.")

    @staticmethod
    def _try_radec_wcs(header):
        try:
            return WCS(header, key="A").celestial
        except Exception:
            return None