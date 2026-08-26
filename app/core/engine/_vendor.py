"""Make the vendored xurdif engine importable.

Adds ``vendor/xurdif`` to sys.path so ``import xurdif`` and
``from alt_models.tinyunet_with_attn3 import ...`` resolve, exactly as the
original scripts expect.
"""
import sys
from pathlib import Path

VENDOR_XURDIF = Path(__file__).resolve().parents[3] / "vendor" / "xurdif"


def ensure_on_path():
    p = str(VENDOR_XURDIF)
    if p not in sys.path:
        sys.path.insert(0, p)
    return VENDOR_XURDIF
