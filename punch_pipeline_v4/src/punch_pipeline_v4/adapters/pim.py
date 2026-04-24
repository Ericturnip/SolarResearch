from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from .base import ProductAdapter
from ..models import LayerFrame, BinnedMap
from ..processing.binning import build_grid_from_wcs, per_bin_median


class PIMAdapter(ProductAdapter):
    product = "PIM"
    default_layer = "Polar_pB"

    raw_layer_index = {
        "polar_m": 0,
        "m": 0,
        "polar_z": 1,
        "z": 1,
        "polar_p": 2,
        "p": 2,
    }

    synthetic_layers = {"polar_pb", "pb", "polar_pb"}

    canonical_raw_names = {
        0: "Polar_M",
        1: "Polar_Z",
        2: "Polar_P",
    }

    def list_layers(self, path: str | Path) -> list[str]:
        return ["Polar_M", "Polar_Z", "Polar_P", "Polar_pB"]

    def make_binned_map(
        self,
        path,
        layer,
        *,
        bin_size_deg: float = 1.0,
        convert_to_s10: bool = False,
    ):
        layer_key = str(layer).strip().lower()

        if layer_key not in {"polar_pb", "pb"}:
            return None

        return self.make_binned_pb(
            path,
            bin_size_deg=bin_size_deg,
            convert_to_s10=convert_to_s10,
        )

    def load_layer(self, path: str | Path, layer: str | None = None) -> LayerFrame:
        requested = layer or self.default_layer
        key = requested.strip().lower()

        path = Path(path).expanduser()

        with fits.open(path, memmap=False) as hdul:
            hdu, header = self._find_science_cube_hdu(hdul)
            cube = np.asarray(hdu.data, dtype=np.float64).squeeze()
            cube_zyx = self._normalize_cube_axis(cube)

            primary_header = hdul[0].header if len(hdul) > 0 else header
            timestamp = self._timestamp_from_headers(header, primary_header)
            solar_wcs = WCS(header).celestial
            radec_wcs = self._try_radec_wcs(header)
            native_unit = header.get("BUNIT") or primary_header.get("BUNIT")

            if key in self.raw_layer_index:
                layer_i = self.raw_layer_index[key]
                data = cube_zyx[layer_i]
                canonical_name = self.canonical_raw_names[layer_i]
                selected_layer_index = layer_i
                computed_from = None
            elif key in {"polar_pb", "pb"}:
                data = self._compute_pb_from_mzp_maps(
                    cube_zyx[0],
                    cube_zyx[1],
                    cube_zyx[2],
                )
                canonical_name = "Polar_pB"
                selected_layer_index = None
                computed_from = ("Polar_M", "Polar_Z", "Polar_P")
            else:
                raise ValueError(
                    f"Unknown PIM layer {requested!r}. "
                    f"Allowed layers: {self.list_layers(path)}"
                )

            metadata = self._metadata(
                path,
                hdu,
                header,
                primary_header,
                cube,
                canonical_name,
                selected_layer_index,
                computed_from,
            )

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

    def make_binned_pb(
        self,
        path,
        *,
        bin_size_deg: float = 1.0,
        convert_to_s10: bool = False,
        s10_coeff: float = 4.5e-16,
    ) -> tuple[BinnedMap, dict]:
        path = Path(path).expanduser()

        with fits.open(path, memmap=False) as hdul:
            hdu, header = self._find_science_cube_hdu(hdul)
            cube = np.asarray(hdu.data, dtype=np.float64).squeeze()
            cube_zyx = self._normalize_cube_axis(cube)

            m = cube_zyx[0]
            z = cube_zyx[1]
            p = cube_zyx[2]

            primary_header = hdul[0].header if len(hdul) > 0 else header
            timestamp = self._timestamp_from_headers(header, primary_header)
            solar_wcs = WCS(header).celestial
            radec_wcs = self._try_radec_wcs(header)
            native_unit = header.get("BUNIT") or primary_header.get("BUNIT")

            grid = build_grid_from_wcs(
                solar_wcs,
                m.shape,
                bin_size_deg=bin_size_deg,
            )

            hpln, hplt = solar_wcs.pixel_to_world_values(
                grid["x_idx"].ravel(),
                grid["y_idx"].ravel(),
            )

            finite_xy = np.isfinite(hpln) & np.isfinite(hplt)

            m_vals = np.asarray(m, dtype=np.float64).ravel()
            z_vals = np.asarray(z, dtype=np.float64).ravel()
            p_vals = np.asarray(p, dtype=np.float64).ravel()

            m_keep = finite_xy & np.isfinite(m_vals)
            z_keep = finite_xy & np.isfinite(z_vals)
            p_keep = finite_xy & np.isfinite(p_vals)

            m_binned = per_bin_median(
                hpln[m_keep],
                hplt[m_keep],
                m_vals[m_keep],
                grid["x_bins"],
                grid["y_bins"],
            )
            z_binned = per_bin_median(
                hpln[z_keep],
                hplt[z_keep],
                z_vals[z_keep],
                grid["x_bins"],
                grid["y_bins"],
            )
            p_binned = per_bin_median(
                hpln[p_keep],
                hplt[p_keep],
                p_vals[p_keep],
                grid["x_bins"],
                grid["y_bins"],
            )

            pb_binned = self._compute_pb_from_mzp_maps(
                m_binned,
                z_binned,
                p_binned,
            )

            if convert_to_s10:
                values = pb_binned / s10_coeff
                unit = "S10"
            else:
                values = pb_binned
                unit = native_unit

            time_map = np.full(
                values.shape,
                timestamp.to_datetime().isoformat(),
                dtype="<U30",
            )
            time_map[~np.isfinite(values)] = ""

            metadata = self._metadata(
                path,
                hdu,
                header,
                primary_header,
                cube,
                "Polar_pB",
                None,
                ("Polar_M", "Polar_Z", "Polar_P"),
            )
            metadata["pim_binning_mode"] = "bin_mzp_then_compute_pb"
            metadata["converted_to_s10_before_output"] = bool(convert_to_s10)
            metadata["s10_coeff"] = s10_coeff if convert_to_s10 else None

        bmap = BinnedMap(
            product=self.product,
            layer_name="Polar_pB",
            values=values,
            hpln_centers=grid["hpln_centers"],
            hplt_centers=grid["hplt_centers"],
            timestamp=timestamp,
            solar_wcs=solar_wcs,
            radec_wcs=radec_wcs,
            time_map=time_map,
            unit=unit,
            metadata=metadata,
        )

        finite = np.isfinite(values)

        diag = {
            "input_shape": tuple(m.shape),
            "m_binned_finite_count": int(np.isfinite(m_binned).sum()),
            "z_binned_finite_count": int(np.isfinite(z_binned).sum()),
            "p_binned_finite_count": int(np.isfinite(p_binned).sum()),
            "binned_shape": tuple(values.shape),
            "binned_finite_count": int(finite.sum()),
            "binned_nan_count": int(np.isnan(values).sum()),
            "binned_zero_count": int((finite & (values == 0.0)).sum()),
            "binned_negative_count": int((finite & (values < 0.0)).sum()),
            "binned_min": float(np.nanmin(values)) if np.any(finite) else np.nan,
            "binned_median": float(np.nanmedian(values)) if np.any(finite) else np.nan,
            "binned_max": float(np.nanmax(values)) if np.any(finite) else np.nan,
            "bin_size_deg": bin_size_deg,
            "converted_to_s10_before_binning": bool(convert_to_s10),
            "pim_binning_mode": "bin_mzp_then_compute_pb",
        }

        return bmap, diag

    @staticmethod
    def _compute_pb_from_mzp_maps(m_map, z_map, p_map):
        finite = np.isfinite(m_map) & np.isfinite(z_map) & np.isfinite(p_map)

        pb = np.full(np.shape(m_map), np.nan, dtype=np.float64)
        if not np.any(finite):
            return pb

        q = (4.0 / 3.0) * z_map[finite] - (2.0 / 3.0) * (
            p_map[finite] + m_map[finite]
        )
        u = (2.0 / np.sqrt(3.0)) * p_map[finite] - (
            2.0 / np.sqrt(3.0)
        ) * m_map[finite]

        pb[finite] = np.sqrt(q * q + u * u)
        return pb

    @staticmethod
    def _normalize_cube_axis(cube: np.ndarray) -> np.ndarray:
        if cube.ndim != 3:
            raise ValueError(f"PIM science data must be 3D; got shape {cube.shape}")

        if cube.shape[0] == 3:
            return cube

        if cube.shape[-1] == 3:
            return np.moveaxis(cube, -1, 0)

        raise ValueError(
            f"Cannot identify PIM layer axis. Expected one axis of length 3; got shape {cube.shape}"
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

        raise ValueError(
            f"Could not find a 3D PIM science cube HDU. HDUs seen: {shapes}"
        )

    @staticmethod
    def _timestamp_from_headers(*headers) -> Time:
        for key in ("DATE-AVG", "DATE-OBS", "DATE-BEG", "DATE-END"):
            for header in headers:
                value = header.get(key) if header is not None else None
                if value:
                    return Time(value, format="isot", scale="utc")

        raise ValueError("PIM file has no DATE-AVG/DATE-OBS/DATE-BEG/DATE-END timestamp.")

    @staticmethod
    def _try_radec_wcs(header):
        try:
            return WCS(header, key="A").celestial
        except Exception:
            return None

    @staticmethod
    def _metadata(
        path,
        hdu,
        header,
        primary_header,
        cube,
        canonical_name,
        selected_layer_index,
        computed_from,
    ) -> dict[str, Any]:
        return {
            "source_path": str(path),
            "filename": path.name,
            "hdu_index": None,
            "date_beg": header.get("DATE-BEG") or primary_header.get("DATE-BEG"),
            "date_obs": header.get("DATE-OBS") or primary_header.get("DATE-OBS"),
            "date_avg": header.get("DATE-AVG") or primary_header.get("DATE-AVG"),
            "date_end": header.get("DATE-END") or primary_header.get("DATE-END"),
            "bunit": header.get("BUNIT") or primary_header.get("BUNIT"),
            "ctype1": header.get("CTYPE1"),
            "ctype2": header.get("CTYPE2"),
            "ctype3": header.get("CTYPE3"),
            "ctype1a": header.get("CTYPE1A"),
            "ctype2a": header.get("CTYPE2A"),
            "ctype3a": header.get("CTYPE3A"),
            "shape": tuple(cube.shape),
            "selected_layer_index": selected_layer_index,
            "selected_layer_name": canonical_name,
            "computed_from": computed_from,
            "obslayr1": header.get("OBSLAYR1"),
            "obslayr2": header.get("OBSLAYR2"),
            "obslayr3": header.get("OBSLAYR3"),
            "obs_mode": header.get("OBS-MODE"),
            "level": header.get("LEVEL") or primary_header.get("LEVEL"),
            "typecode": header.get("TYPECODE") or primary_header.get("TYPECODE"),
        }