from __future__ import annotations

from pathlib import Path
from .base import ProductAdapter
from ..models import LayerFrame


class CAMAdapter(ProductAdapter):
    product = "CAM"

    def load_layer(self, path: str | Path, layer: str | None = None) -> LayerFrame:
        """TODO: implement CAM FITS loading.

        This should return a LayerFrame with:
        - product="CAM"
        - layer_name
        - 2D data array
        - timestamp as astropy.time.Time
        - solar_wcs
        - radec_wcs if available
        - native_unit / BUNIT if available
        """
        raise NotImplementedError("Fill CAM adapter with product-specific FITS logic.")

    def list_layers(self, path: str | Path) -> list[str]:
        raise NotImplementedError("Fill CAM adapter layer discovery.")
