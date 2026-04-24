from __future__ import annotations

from .cim import CIMAdapter
from .ctm import CTMAdapter
from .pim import PIMAdapter
from .ptm import PTMAdapter
from .pam import PAMAdapter
from .cam import CAMAdapter


def get_adapter(product: str):
    p = product.upper()
    adapters = {
        "CIM": CIMAdapter,
        "CTM": CTMAdapter,
        "PIM": PIMAdapter,
        "PTM": PTMAdapter,
        "PAM": PAMAdapter,
        "CAM": CAMAdapter,
    }
    if p not in adapters:
        raise ValueError(f"Unknown product {product}. Known: {sorted(adapters)}")
    return adapters[p]()
