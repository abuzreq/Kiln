"""Run a smoke test in a throwaway workspace instead of your real one.

Import this before anything from ``app``: the workspace is chosen once, when
``app.core.config`` is first imported. It points ``KILN_WORKSPACE`` at a fresh
temporary folder and deletes that folder on exit. If ``KILN_WORKSPACE`` is
already set, that choice is left alone.
"""
import atexit
import os
import shutil
import tempfile

if not os.environ.get("KILN_WORKSPACE"):
    _root = tempfile.mkdtemp(prefix="kiln-smoke-")
    os.environ["KILN_WORKSPACE"] = _root
    atexit.register(shutil.rmtree, _root, True)
