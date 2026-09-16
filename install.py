"""Kiln installer.

Creates an isolated virtual environment, installs Python dependencies, optionally
builds the React frontend, and (with --launch) starts the app. Invoked by the
per-platform launch scripts (kiln.bat / kiln.sh / kiln.command).

Later launches skip pip when a small fingerprint of the requirements files still
matches and a cheap import probe succeeds — no torch import, no pip resolver.
"""
import sys

# Version guard first, before anything below is evaluated. Kiln needs 3.10+
# (app/core/config.py uses PEP 604 `X | None` annotations, and this file uses
# `list[Path]`), and both fail at *evaluation* time with a bare TypeError that
# gives a user no idea what went wrong. This file must stay parseable on old
# interpreters for the message to ever be seen -- keep the syntax here plain.
if sys.version_info < (3, 10):
    sys.stderr.write(
        "\nKiln needs Python 3.10 or newer.\n"
        "You are running Python %d.%d from:\n  %s\n\n"
        "Install a newer Python from https://www.python.org/downloads/\n"
        "then run this launcher again.\n\n"
        % (sys.version_info[0], sys.version_info[1], sys.executable)
    )
    raise SystemExit(1)

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
FRONTEND = ROOT / "app" / "frontend"
STAMP_FILE = VENV_DIR / "kiln-deps.json"

# The Kiln-only subset of the vendored engine's requirements. Its own
# requirements.txt additionally pulls gradio for the standalone reference apps,
# which Kiln never imports -- see the header of requirements-kiln.txt.
VENDOR_REQ = ROOT / "vendor" / "xurdif" / "requirements-kiln.txt"

# Cheap modules (importing torch here would add seconds on every launch).
_PROBE_MODULES = ("flask", "flask_cors", "PIL", "numpy", "cv2", "einops")


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd, **kw)


def _venv_help() -> str:
    """What to install when the standard library cannot build a venv.

    Debian and Ubuntu ship Python without ensurepip and without venv's package
    metadata, so ``python3 -m venv`` fails there on a machine that otherwise has
    a perfectly good Python. It is the most common first-run failure on Linux,
    and the raw traceback names none of this.
    """
    import shutil

    ver = "%d.%d" % (sys.version_info[0], sys.version_info[1])
    if platform.system() != "Linux":
        return "Reinstall Python from https://www.python.org/downloads/ and try again."
    if shutil.which("apt-get") or shutil.which("apt"):
        return ("On Debian/Ubuntu the venv module is a separate package:\n"
                "    sudo apt install python3-venv python3-pip\n"
                "  (if that is not enough, try python" + ver + "-venv)")
    if shutil.which("dnf") or shutil.which("yum"):
        return "On Fedora/RHEL:\n    sudo dnf install python3-pip"
    if shutil.which("pacman"):
        return "On Arch:\n    sudo pacman -S python-pip"
    return "Install your distribution's python3-venv / python3-pip packages and try again."


def ensure_venv():
    if venv_python().exists():
        print("Virtual environment already present.")
        return
    print("Creating virtual environment in .venv ...")
    try:
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "\nCould not create the virtual environment in %s.\n"
            "  %s: %s\n\n%s\n" % (VENV_DIR, type(e).__name__, e, _venv_help())
        )


def _req_files() -> list[Path]:
    paths = [ROOT / "requirements.txt", VENDOR_REQ]
    return [p for p in paths if p.exists()]


def _fingerprint() -> str:
    """Hash of requirement files + interpreter version. Fast; no subprocess.

    Major.minor, not the full ``sys.version``: a venv is tied to the minor
    version, but the build string moves on every patch release, and on a distro
    that updates Python regularly that meant a full pip pass after an upgrade
    that changed nothing a wheel cares about.
    """
    h = hashlib.sha256()
    h.update(("%d.%d" % (sys.version_info[0], sys.version_info[1])).encode())
    h.update(platform.system().encode())
    for p in _req_files():
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _read_stamp() -> dict:
    try:
        return json.loads(STAMP_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write_stamp(torch_build: str | None = None, driver=None):
    STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAMP_FILE.write_text(json.dumps({
        "fingerprint": _fingerprint(),
        "python": sys.version.split()[0],
        # Which CUDA the installed torch was built for, and the driver it was
        # chosen for. Launch compares them without importing torch, so a venv
        # carried to another machine (or a driver rollback) is caught.
        "torch_cuda": torch_build,
        "driver": driver,
        "updated_at": time.time(),
    }, indent=2), encoding="utf-8")


def warn_if_torch_mismatch():
    """One cheap line at launch when the installed build cannot run here.

    Reads the stamp and asks nvidia-smi for the driver: no torch import, so it
    costs nothing on a healthy install.
    """
    stamp = _read_stamp()
    build = stamp.get("torch_cuda")
    if not build or not _nvidia_gpu_present():
        return
    driver = cuda_pick.driver_version()
    if cuda_pick.build_is_runnable(build, driver) is not False:
        return
    print(
        f"\nNOTE: the PyTorch installed here is built for CUDA {build}, which this "
        f"machine's driver ({driver:g}) cannot run.\n"
        "      Kiln will fall back to CPU, and training needs a GPU. Fix it with:\n"
        f"      {sys.executable} install.py --fix-torch\n"
    )


def _probe_imports(py: Path) -> bool:
    """One short subprocess: import light packages. Does not import torch."""
    code = (
        "import importlib.util, sys\n"
        "mods = " + repr(_PROBE_MODULES) + "\n"
        "missing = [m for m in mods if importlib.util.find_spec(m) is None]\n"
        "sys.exit(1 if missing else 0)\n"
    )
    try:
        subprocess.check_call(
            [str(py), "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def _torch_dist_present(py: Path) -> bool:
    """True if a torch wheel is installed, without importing it."""
    code = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
    try:
        subprocess.check_call(
            [str(py), "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def deps_satisfied() -> bool:
    """True when the venv looks current — skip pip. Typically tens of milliseconds."""
    py = venv_python()
    if not py.exists():
        return False
    stamp = _read_stamp()
    if stamp.get("fingerprint") != _fingerprint():
        return False
    if not _probe_imports(py):
        return False
    if not _torch_dist_present(py):
        return False
    return True


# PyPI's default torch wheel is CPU-only on Windows, and on Linux it is built for
# the newest CUDA there is -- which an older driver cannot run. Either way a
# plain requirements install can leave a machine with a torch that cannot train.
# Kiln's engine is CUDA-only, so pick the build this driver *can* run and get it
# in first. utils/cuda.py holds the driver -> build table; it is stdlib-only, so
# importing it here, before any dependency exists, is safe.
sys.path.insert(0, str(ROOT))
from utils import cuda as cuda_pick  # noqa: E402


def _nvidia_gpu_present() -> bool:
    """True when nvidia-smi reports at least one GPU. Cheap, no torch import."""
    import shutil

    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=6,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def _torch_build(py: Path) -> tuple:
    """(version, CUDA build, usable) for the torch installed in ``py``."""
    code = ("import json, torch;"
            "print(json.dumps([torch.__version__, torch.version.cuda, "
            "torch.cuda.is_available()]))")
    try:
        out = subprocess.check_output(
            [str(py), "-c", code], stderr=subprocess.DEVNULL, text=True, timeout=300)
        return tuple(json.loads(out.strip().splitlines()[-1]))
    except Exception:  # noqa: BLE001
        return (None, None, False)


def install_cuda_torch(py: Path, force: bool = False) -> bool:
    """Install the CUDA torch/torchvision build this machine's driver can run.

    Returns True when a usable CUDA build is in place afterwards.
    """
    if not _nvidia_gpu_present():
        print("No NVIDIA GPU detected — installing the default (CPU) PyTorch build.")
        return False

    driver = cuda_pick.driver_version()
    url, cuda, _name = cuda_pick.build_for(driver)
    version, build, usable = (_torch_build(py) if _torch_dist_present(py)
                              else (None, None, False))

    if usable and not force:
        print(f"CUDA-enabled PyTorch already installed ({version}, CUDA {build}) "
              "— leaving it alone.")
        return True
    if url is None:
        print(f"NVIDIA driver {driver} is older than any current PyTorch build "
              "supports (450 and up). Kiln will fall back to CPU.")
        return False

    drv = f"{driver:g}" if driver is not None else "unknown"
    if version and build and cuda_pick.build_is_runnable(build, driver) is False:
        # PyPI's default wheel on a driver that predates it. --upgrade alone
        # would leave it in place: it is *newer*, just unusable here.
        print(f"PyTorch {version} here is built for CUDA {build}, which driver {drv} "
              f"cannot run. Replacing it with the CUDA {cuda} build.")
    print(
        f"\nNVIDIA driver {drv}: installing the CUDA {cuda} build of PyTorch from\n"
        f"  {url}\n"
        "This download is large (~2.5 GB) but Kiln's engine cannot train or\n"
        "sample without it.\n"
    )
    cmd = [str(py), "-m", "pip", "install", "--upgrade",
           "torch", "torchvision", "--index-url", url]
    if version:
        cmd.insert(5, "--force-reinstall")
    try:
        run(cmd)
    except Exception as e:  # noqa: BLE001
        joined = " ".join(cmd)
        print(
            f"WARNING: that install failed ({e}).\n"
            "Kiln will fall back to CPU. Retry manually with:\n"
            f"  {joined}\n"
            "If pip found no matching wheel, this Python is probably newer than that "
            "CUDA build supports: install Kiln under an older Python, or update the "
            "driver so a newer build can be used."
        )
        return False
    return _torch_is_cuda(py)


def _torch_constraints(py: Path):
    """Pin the torch just chosen so the requirements pass cannot swap it.

    ``torch>=2.0`` in requirements.txt is satisfied by whatever is installed, but
    a dependency asking for a newer torch would pull PyPI's default wheel -- the
    newest CUDA build -- straight over a deliberately chosen older one.
    """
    code = ("import json, importlib.metadata as m\n"
            "out = {}\n"
            "for n in ('torch', 'torchvision'):\n"
            "    try:\n"
            "        out[n] = m.version(n)\n"
            "    except Exception:\n"
            "        pass\n"
            "print(json.dumps(out))")
    try:
        out = subprocess.check_output(
            [str(py), "-c", code], stderr=subprocess.DEVNULL, text=True, timeout=120)
        versions = json.loads(out.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return None
    if not versions:
        return None
    path = VENV_DIR / "kiln-torch-constraints.txt"
    path.write_text("".join(f"{n}=={v}\n" for n, v in versions.items()), encoding="utf-8")
    return path


def _torch_is_cuda(py) -> bool:
    try:
        out = subprocess.check_output(
            [str(py), "-c", "import torch;print(torch.cuda.is_available())"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out == "True"
    except Exception:  # noqa: BLE001
        return False


def install_requirements(force: bool = False):
    py = venv_python()
    if not force and deps_satisfied():
        print("Dependencies already satisfied — skipping pip.")
        return

    # Existing venv from before fingerprinting: if imports already work, just stamp
    # and skip pip. Changing requirements.txt will change the hash and reinstall.
    if (
        not force
        and py.exists()
        and not _read_stamp()
        and _probe_imports(py)
        and _torch_dist_present(py)
    ):
        print("Existing install looks complete — skipping pip.")
        _write_stamp(torch_build=_torch_build(py)[1], driver=cuda_pick.driver_version())
        return

    # Only bump pip on a real install; not on every launch.
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    # Get the right torch in first: requirements.txt only asks for `torch>=2.0`,
    # which pip would satisfy with a CPU-only wheel and then leave alone.
    install_cuda_torch(py)
    # ... and hold it there: -c pins that exact build through both passes.
    cons = _torch_constraints(py)
    pin = ["-c", str(cons)] if cons else []

    run([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), *pin])

    if VENDOR_REQ.exists():
        print("Installing vendored xurdif engine requirements ...")
        run([str(py), "-m", "pip", "install", "-r", str(VENDOR_REQ), *pin])

    _check_native_libs(py)

    # Belt and braces: if anything above still managed to replace torch, put the
    # right build back before stamping the install as good.
    if _nvidia_gpu_present() and not _torch_is_cuda(py):
        print("PyTorch cannot use the GPU after the dependency install — repairing.")
        install_cuda_torch(py, force=True)

    _write_stamp(torch_build=_torch_build(py)[1], driver=cuda_pick.driver_version())

    if _torch_is_cuda(py):
        print("PyTorch reports CUDA is available — GPU training/sampling is ready.")
    elif _nvidia_gpu_present():
        version, build, _ = _torch_build(py)
        report = cuda_pick.diagnose(version, build, cuda_pick.driver_version(), str(py))
        print(
            "WARNING: an NVIDIA GPU is present but PyTorch in the venv cannot use it.\n"
            "Training and sampling will not work.\n"
            f"  {report['hint']}"
        )
    else:
        print(
            "No NVIDIA GPU detected: dataset prep, model inspection, bending and "
            "merging work on CPU; training and sampling need CUDA."
        )


def _check_native_libs(py: Path):
    """One real import of OpenCV, once, after a fresh install.

    The launch probe only asks whether packages are *present* (find_spec, no
    import) so that startup stays fast. That cannot see a missing system
    library: opencv-python links libGL, which minimal and server Linux images do
    not ship, and the failure would otherwise surface much later as a traceback
    in the middle of preparing a dataset.
    """
    out = subprocess.run([str(py), "-c", "import cv2"], capture_output=True, text=True)
    if out.returncode == 0:
        return
    err = (out.stderr or "").strip().splitlines()
    last = err[-1] if err else "unknown error"
    print("\nWARNING: OpenCV is installed but does not import:\n  %s" % last)
    if "libGL" in last or "libgl" in last or "libglib" in last.lower():
        print(
            "  It needs system graphics libraries that this image does not carry.\n"
            "  On Debian/Ubuntu:\n"
            "    sudo apt install libgl1 libglib2.0-0\n"
            "  Dataset prep from video and the post-processing chain need it."
        )


def build_frontend():
    build_dir = FRONTEND / "build"
    if build_dir.exists() and any(build_dir.iterdir()):
        print("Frontend build already present; skipping (delete app/frontend/build to rebuild).")
        return
    npm = "npm.cmd" if platform.system() == "Windows" else "npm"
    try:
        run([npm, "--version"])
    except Exception:  # noqa: BLE001
        print(
            "npm not found. The prebuilt frontend in app/frontend/build will be served "
            "as-is. Install Node 18+ and run `npm install && npm run build` in "
            "app/frontend to rebuild."
        )
        return
    try:
        run([npm, "install"], cwd=str(FRONTEND))
        run([npm, "run", "build"], cwd=str(FRONTEND))
    except Exception as e:  # noqa: BLE001
        print(f"Frontend build failed ({e}); the committed build/ will be served instead.")


def launch(extra_args):
    py = venv_python()
    env = os.environ.copy()
    run([str(py), str(ROOT / "start.py"), *extra_args], env=env)


def main():
    # pip writes straight to the console, while our own print() goes through a
    # block buffer as soon as output is redirected to a file -- so the progress
    # messages between pip runs ("installing the CUDA 12.6 build ...") landed out
    # of order, or vanished entirely when a run was interrupted. Line buffering
    # costs nothing here and keeps the narration next to what it describes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="start Kiln after install")
    ap.add_argument("--skip-frontend", action="store_true")
    ap.add_argument("--reinstall", action="store_true", help="run pip even if the fingerprint matches")
    ap.add_argument("--fix-torch", action="store_true",
                    help="reinstall the PyTorch build this machine's driver can run, then exit")
    args, extra = ap.parse_known_args()

    ensure_venv()
    if args.fix_torch:
        py = venv_python()
        ok = install_cuda_torch(py, force=True)
        version, build, _ = _torch_build(py)
        state = "available" if ok else "NOT available"
        print(f"\nPyTorch {version} (CUDA {build}) — GPU is {state} in {py}")
        if not ok:
            report = cuda_pick.diagnose(version, build, cuda_pick.driver_version(), str(py))
            print(f"  {report['hint']}")
        if not args.launch:
            return

    install_requirements(force=args.reinstall)
    if not args.skip_frontend:
        build_frontend()
    warn_if_torch_mismatch()
    if args.launch:
        launch(extra)
    else:
        print("\nInstall complete. Start Kiln with:  python start.py")


if __name__ == "__main__":
    main()
