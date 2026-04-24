from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from ..models import LayerFrame


class ProductAdapter:
    product: str
    default_layer: str

    def list_layers(self, path):
        raise NotImplementedError

    def load_layer(self, path, layer=None):
        raise NotImplementedError

    def make_binned_map(
        self,
        path,
        layer,
        *,
        bin_size_deg: float = 1.0,
        convert_to_s10: bool = False,
    ):
        """
        Optional adapter hook for products whose correct science logic requires
        custom binning.

        Default behavior:
        - return None
        - caller falls back to load_layer() + median_bin_layer_frame()

        Example:
        PIM Polar_pB needs:
        M/Z/P -> median-bin separately -> compute pB
        """
        return None
