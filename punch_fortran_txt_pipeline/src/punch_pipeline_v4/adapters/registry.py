from __future__ import annotations

from .cim import CIMAdapter
from .ctm import CTMAdapter
from .pam import PAMAdapter
from .pim import PIMAdapter
from .ptm import PTMAdapter


def get_adapter(product: str):
    p = product.upper()
    adapters = {
        "CIM": CIMAdapter,
        "CTM": CTMAdapter,
        "PAM": PAMAdapter,
        "PIM": PIMAdapter,
        "PTM": PTMAdapter,
    }
    if p not in adapters:
        raise ValueError(f"Unknown product {product}. Known: {sorted(adapters)}")
    return adapters[p]()
