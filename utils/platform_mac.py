"""What kind of Mac this is, without importing torch.

Kiln's GPU path on a Mac is Apple's Metal backend (``mps``), which exists only
on Apple Silicon. Two cases look alike from Python and are not: an Intel Mac
has no such GPU at all, while an Apple Silicon Mac running an Intel build of
Python under Rosetta has one that the x86_64 torch wheel it installs can never
see. ``platform.machine()`` says "x86_64" for both, so the hardware has to be
asked separately.

Stdlib only, and importable before any dependency is installed: ``install.py``
uses it to choose what to install, and the app to explain what it found.
"""
import functools
import platform
import subprocess
import sys


def is_mac() -> bool:
    return sys.platform == "darwin"


def _sysctl(name: str) -> str:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


@functools.lru_cache(maxsize=None)
def hardware_is_arm64() -> bool:
    """Is the machine Apple Silicon, whatever this Python was built for?"""
    if not is_mac():
        return False
    return platform.machine() == "arm64" or _sysctl("hw.optional.arm64") == "1"


def is_apple_silicon() -> bool:
    """An Apple Silicon Mac running a native (arm64) Python: the MPS case."""
    return is_mac() and platform.machine() == "arm64"


def under_rosetta() -> bool:
    """An Intel build of Python on Apple Silicon. Its torch cannot use the GPU."""
    return is_mac() and platform.machine() != "arm64" and hardware_is_arm64()


@functools.lru_cache(maxsize=None)   # the device badge asks every few seconds
def chip_name() -> str:
    """"Apple M2 Pro" and the like; empty when it cannot be read."""
    return _sysctl("machdep.cpu.brand_string") if is_mac() else ""
