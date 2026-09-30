"""Which PyTorch build an NVIDIA driver can actually run.

PyTorch ships one build per CUDA generation, and PyPI's default wheel tracks the
newest one -- today that is CUDA 13 on Linux, and CPU-only on Windows. A CUDA
build only runs on a driver new enough for it, so installing the default on an
older driver leaves a torch that cannot see a perfectly good GPU. In Kiln that
surfaced as "PyTorch is built for CUDA 13.0 ... check the driver version", which
reads as though Kiln wanted CUDA 13. It does not: it wants whichever build the
driver in the machine can run.

The minimums below are NVIDIA's minor-version-compatibility table from the CUDA
Toolkit release notes: a CUDA 13.x build needs driver >= 580, 12.x >= 525, and
11.x >= 450. Within a family any minor version works, which is why one index per
family is enough.

Stdlib only, and importable before any dependency is installed: ``install.py``
uses it to choose a wheel, and ``app/core/config.py`` to explain a mismatch.
"""
import shutil
import subprocess

INDEX = "https://download.pytorch.org/whl/{}"

# (minimum driver, index name, CUDA version it provides), newest first.
# RTX 50-series cards need cu128 or newer and require a 570+ driver anyway, so
# choosing by driver gets those right without a card lookup.
BUILDS = (
    (580.0, "cu130", "13.0"),
    (575.51, "cu129", "12.9"),
    (570.0, "cu128", "12.8"),
    (525.60, "cu126", "12.6"),
    (450.80, "cu118", "11.8"),
)

# What to install when nvidia-smi is present but its version will not parse:
# the CUDA 12 build runs on every driver from 525 up, which is every driver
# still in support.
FALLBACK = BUILDS[3]

# Lowest driver each CUDA family runs on, for explaining an existing build.
FAMILY_MIN = {13: 580.0, 12: 525.60, 11: 450.80}


def driver_version() -> float | None:
    """Major.minor of the installed NVIDIA driver (535.23), or None.

    ``--query-gpu`` rather than the ``nvidia-smi`` banner: it is one short line,
    cheap enough to call from a request handler, and stable across versions.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=6,
        )
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0 or not (out.stdout or "").strip():
        return None
    parts = out.stdout.strip().splitlines()[0].strip().split(".")
    try:
        return float(f"{parts[0]}.{parts[1]}") if len(parts) >= 2 else float(parts[0])
    except (ValueError, IndexError):
        return None


def build_for(driver: float | None) -> tuple[str | None, str | None, str | None]:
    """(index url, CUDA version, index name) to install for this driver.

    ``(None, None, None)`` when the driver predates every current build; the
    caller should then say so rather than installing something unusable.
    """
    if driver is None:
        _, name, cuda = FALLBACK
        return INDEX.format(name), cuda, name
    for minimum, name, cuda in BUILDS:
        if driver >= minimum:
            return INDEX.format(name), cuda, name
    return None, None, None


def min_driver_for(torch_cuda: str | None) -> float | None:
    """Lowest driver that can run a torch built for ``torch_cuda`` ("13.0")."""
    if not torch_cuda:
        return None
    try:
        return FAMILY_MIN.get(int(str(torch_cuda).split(".")[0]))
    except ValueError:
        return None


def build_is_runnable(torch_cuda: str | None, driver: float | None) -> bool | None:
    """Can this driver run a torch built for ``torch_cuda``? None if unknown."""
    need = min_driver_for(torch_cuda)
    if need is None or driver is None:
        return None
    return driver >= need


def fix_command(driver: float | None, python: str = "python") -> str | None:
    """The pip line that installs the build this driver can run."""
    url, _cuda, _name = build_for(driver)
    if url is None:
        return None
    return (f"{python} -m pip install --force-reinstall torch torchvision "
            f"--index-url {url}")


def diagnose(torch_version, torch_cuda, driver, python: str = "python",
             gpu_present: bool = True) -> dict:
    """Why CUDA is unavailable, and the one command that fixes it.

    Returns ``{"reason", "hint"}``. The reasons are machine-readable for the UI:
    ``cpu_build`` (no CUDA in this wheel), ``build_too_new`` (the wheel needs a
    newer driver than this machine has), ``driver_too_old`` (no current build
    runs on this driver), ``driver`` (the pairing looks fine, so something else
    went wrong).
    """
    drv = f"{driver:g}" if driver is not None else "unknown"
    cmd = fix_command(driver, python)
    _url, cuda, _name = build_for(driver)

    if not torch_cuda:
        where = " — an NVIDIA GPU is present but unusable." if gpu_present else "."
        hint = f"PyTorch {torch_version} is a CPU-only build{where}"
        if cmd:
            hint += (f" This machine's driver ({drv}) runs the CUDA {cuda} build: {cmd}"
                     "  — or: python install.py --fix-torch")
        return {"reason": "cpu_build", "hint": hint}

    if cmd is None:
        return {"reason": "driver_too_old", "hint": (
            f"NVIDIA driver {drv} is older than 450, which no current PyTorch build "
            "supports. Updating the driver is the only fix here.")}

    if build_is_runnable(torch_cuda, driver) is False:
        need = min_driver_for(torch_cuda)
        return {"reason": "build_too_new", "hint": (
            f"PyTorch {torch_version} is built for CUDA {torch_cuda}, which needs NVIDIA "
            f"driver {need:g} or newer; this machine has {drv}. Kiln does not need CUDA "
            f"{torch_cuda} — no driver update required. Install the build this driver runs "
            f"(CUDA {cuda}):  {cmd}  — or: python install.py --fix-torch")}

    return {"reason": "driver", "hint": (
        f"PyTorch {torch_version} (CUDA {torch_cuda}) should run on driver {drv}, but CUDA "
        "did not initialise. A reboot after a driver update usually clears this; otherwise "
        f"reinstall the matching build:  {cmd}")}
