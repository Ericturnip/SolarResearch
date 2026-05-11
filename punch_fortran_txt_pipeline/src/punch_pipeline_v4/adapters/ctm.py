from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from .base import ProductAdapter
from ..models import LayerFrame


class CTMAdapter(ProductAdapter):
    """Adapter for PUNCH L3 CTM files.

    CTM is treated as a single-layer product. The adapter normalizes the FITS
    image into one 2D layer named ``brightness`` and leaves all scientific
    processing to the shared v4 processing code.

    The sample CTM file is a compressed image HDU with:
    - ZNAXIS1 = 4096
    - ZNAXIS2 = 4096
    - CTYPE1/2 = HPLN-ARC/HPLT-ARC
    - alternate A WCS = RA---ARC/DEC--ARC
    - BUNIT = native radiance unit, not S10
    """

    product = "CTM"
    default_layer = "brightness"
    aliases = {"brightness", "ctm", "data", "image", "unpolarized", "b"}

    def list_layers(self, path: str | Path) -> list[str]:
        """CTM has one science image layer."""
        return [self.default_layer]

    def load_layer(self, path: str | Path, layer: str | None = None) -> LayerFrame:
        requested = (layer or self.default_layer).strip()
        if requested.lower() not in self.aliases:
            raise ValueError(
                f"CTM has one layer named '{self.default_layer}'. "
                f"Got layer={requested!r}. Accepted aliases: {sorted(self.aliases)}"
            )

        path = Path(path).expanduser()
        with fits.open(path, memmap=False) as hdul:
            hdu, header = self._find_science_image_hdu(hdul)
            data = np.asarray(hdu.data, dtype=np.float64)

            if data.ndim != 2:
                raise ValueError(f"CTM science image must be 2D; got shape {data.shape} from {path}")

            # Prefer the science-image header, but fall back to primary for time metadata.
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
                "ctype1a": header.get("CTYPE1A"),
                "ctype2a": header.get("CTYPE2A"),
                "shape": tuple(data.shape),
                "level": header.get("LEVEL") or primary_header.get("LEVEL"),
                "typecode": header.get("TYPECODE") or primary_header.get("TYPECODE"),
            }

        return LayerFrame(
            product=self.product,
            layer_name=self.default_layer,
            data=data,
            timestamp=timestamp,
            solar_wcs=solar_wcs,
            radec_wcs=radec_wcs,
            native_unit=native_unit,
            metadata=metadata,
        )

    @staticmethod
    def _find_science_image_hdu(hdul: fits.HDUList):
        """Return the first HDU that Astropy exposes as a 2D image.

        PUNCH CTM files may be FITS compressed images. Astropy should expose the
        compressed image extension as an HDU whose ``data`` is a 2D ndarray. We
        intentionally avoid decoding the compression ourselves here.
        """
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            if isinstance(data, np.ndarray) and data.ndim == 2:
                return hdu, hdu.header

        shapes = []
        for i, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            shapes.append((i, type(hdu).__name__, None if data is None else getattr(data, "shape", None)))
        raise ValueError(f"Could not find a 2D CTM science image HDU. HDUs seen: {shapes}")

    @staticmethod
    def _timestamp_from_headers(*headers) -> Time:
        for key in ("DATE-AVG", "DATE-OBS", "DATE-BEG", "DATE-END"):
            for header in headers:
                value = header.get(key) if header is not None else None
                if value:
                    return Time(value, format="isot", scale="utc")
        raise ValueError("CTM file has no DATE-AVG/DATE-OBS/DATE-BEG/DATE-END timestamp.")

    @staticmethod
    def _try_radec_wcs(header):
        try:
            return WCS(header, key="A").celestial
        except Exception:
            return None
