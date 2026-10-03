"""Kiln application package."""
import os
import sys

__version__ = "0.1.0"

# On Apple Silicon an op the Metal backend lacks raises unless this is set, and
# it is read once, when torch is first imported -- which happens after this
# package is. Falling back to the CPU for that one op is slower but finishes.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
