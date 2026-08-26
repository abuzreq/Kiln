"""Kiln installer.

Creates an isolated virtual environment, installs Python dependencies, optionally
builds the React frontend, and (with --launch) starts the app. Invoked by the
per-platform launch scripts (kiln.bat / kiln.sh / kiln.command).

Later launches skip pip when a small fingerprint of the requirements files still
matches and a cheap import probe succeeds — no torch import, no pip resolver.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
FRONTEND = ROOT / "app" / "frontend"
STAMP_FILE = VENV_DIR / "kiln-deps.json"

# Cheap modules (importing torch here would add seconds on every launch).
_PROBE_MODULES = ("flask", "flask_cors", "PIL", "numpy", "cv2", "einops")


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd, **kw)


def ensure_venv():
    if not venv_python().exists():
        print("Creating virtual environment in .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print("Virtual environment already present.")


def _req_files() -> list[Path]:
    paths = [ROOT / "requirements.txt", ROOT / "vendor" / "xurdif" / "requirements.txt"]
    return [p for p in paths if p.exists()]


def _fingerprint() -> str:
    """Hash of requirement files + interpreter version. Fast; no subprocess."""
    h = hashlib.sha256()
    h.update(sys.version.encode())
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


def _write_stamp():
    STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAMP_FILE.write_text(json.dumps({
        "fingerprint": _fingerprint(),
        "python": sys.version.split()[0],
        "updated_at": time.time(),
    }, indent=2), encoding="utf-8")


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


# PyPI's default torch wheel is CPU-only on Windows (and CUDA-less on some
# Linux setups), so a plain requirements install silently yields a torch that
# cannot train. Kiln's engine is CUDA-only, so we install the right wheel up
# front whenever the machine actually has an NVIDIA GPU.
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"


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


def install_cuda_torch(py: Path) -> bool:
    """Install CUDA torch/torchvision if this machine has a GPU and lacks them.

    Returns True when a CUDA build is in place afterwards.
    """
    if not _nvidia_gpu_present():
        print("No NVIDIA GPU detected — installing the default (CPU) PyTorch build.")
        return False
    if _torch_dist_present(py) and _torch_is_cuda(py):
        print("CUDA-enabled PyTorch already installed — leaving it alone.")
        return True

    print(
        "\nNVIDIA GPU detected. Installing a CUDA build of PyTorch from\n"
        f"  {CUDA_INDEX_URL}\n"
        "This download is large (~2.5 GB) but Kiln's engine cannot train or\n"
        "sample without it.\n"
    )
    try:
        run([
            str(py), "-m", "pip", "install", "--upgrade",
            "torch", "torchvision", "--index-url", CUDA_INDEX_URL,
        ])
    except Exception as e:  # noqa: BLE001
        print(
            f"WARNING: CUDA PyTorch install failed ({e}).\n"
            "Kiln will fall back to CPU. Retry manually with:\n"
            f"  {py} -m pip install torch torchvision --index-url {CUDA_INDEX_URL}"
        )
        return False
    return _torch_is_cuda(py)


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
        _write_stamp()
        return

    # Only bump pip on a real install; not on every launch.
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    # Get the right torch in first: requirements.txt only asks for `torch>=2.0`,
    # which pip would satisfy with a CPU-only wheel and then leave alone.
    install_cuda_torch(py)

    run([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])

    vendor_req = ROOT / "vendor" / "xurdif" / "requirements.txt"
    if vendor_req.exists():
        print("Installing vendored xurdif engine requirements ...")
        run([str(py), "-m", "pip", "install", "-r", str(vendor_req)])

    _write_stamp()

    if _torch_is_cuda(py):
        print("PyTorch reports CUDA is available — GPU training/sampling is ready.")
    elif _nvidia_gpu_present():
        print(
            "WARNING: an NVIDIA GPU is present but PyTorch in the venv is CPU-only.\n"
            "Training and sampling will not work. Fix with:\n"
            f"  {py} -m pip install --upgrade torch torchvision "
            f"--index-url {CUDA_INDEX_URL}"
        )
    else:
        print(
            "No NVIDIA GPU detected: dataset prep, model inspection, bending and "
            "merging work on CPU; training and sampling need CUDA."
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="start Kiln after install")
    ap.add_argument("--skip-frontend", action="store_true")
    ap.add_argument("--reinstall", action="store_true", help="run pip even if the fingerprint matches")
    args, extra = ap.parse_known_args()

    ensure_venv()
    install_requirements(force=args.reinstall)
    if not args.skip_frontend:
        build_frontend()
    if args.launch:
        launch(extra)
    else:
        print("\nInstall complete. Start Kiln with:  python start.py")


if __name__ == "__main__":
    main()
