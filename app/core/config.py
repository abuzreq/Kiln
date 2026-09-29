"""Workspace configuration and paths for Kiln.

Kiln keeps everything a user generates (datasets, models, bends, presets,
captures, library entries) inside a single workspace folder, defaulting to ``~/kiln``.
The chosen folder is remembered in a small prefs file under the user config dir.
"""
import json
import os
from pathlib import Path

from utils.logger import get_logger

log = get_logger("config")

APP_NAME = "kiln"

# The install directory. Kiln also scans models/ here, so the UI needs to be able
# to name and open those folders even though they are not part of the workspace.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8777


def _prefs_path() -> Path:
    base = os.environ.get("APPDATA") or os.path.join(str(Path.home()), ".config")
    return Path(base) / APP_NAME / "config.json"


def _default_workspace() -> Path:
    return Path.home() / APP_NAME


class Workspace:
    """Resolves and lays out the Kiln workspace directory tree."""

    SUBDIRS = (
        "source",
        "datasets",
        "runs",
        "models",
        "captures",
        "assets",
        "sweeps",
        "cache",
        "library/bends",
        "library/recipes",
        "library/presets",
    )

    def __init__(self, root: Path | None = None):
        self.prefs_file = _prefs_path()
        self.root = Path(root) if root else self._load_root()
        self.ensure()

    def _load_root(self) -> Path:
        env = os.environ.get("KILN_WORKSPACE")
        if env:
            return Path(env)
        try:
            if self.prefs_file.exists():
                data = json.loads(self.prefs_file.read_text(encoding="utf-8"))
                if data.get("workspace"):
                    return Path(data["workspace"])
        except Exception as e:  # noqa: BLE001
            log.warning("could not read prefs: %s", e)
        return _default_workspace()

    def ensure(self):
        for sub in ("",) + self.SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def set_root(self, new_root: str | os.PathLike):
        self.root = Path(new_root)
        self.ensure()
        self.prefs_file.parent.mkdir(parents=True, exist_ok=True)
        self.prefs_file.write_text(
            json.dumps({"workspace": str(self.root)}, indent=2), encoding="utf-8"
        )
        log.info("workspace set to %s", self.root)

    # --- path helpers -------------------------------------------------
    @property
    def source(self) -> Path:
        return self.root / "source"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def projects(self) -> Path:
        """Legacy project folders; still scanned for existing data."""
        return self.root / "projects"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def captures(self) -> Path:
        return self.root / "captures"

    @property
    def assets(self) -> Path:
        """Images the user brought in to work from. Separate from captures: a
        capture is something Kiln made, an asset is something the user chose to
        keep at hand -- so it can be placed on a layer again without hunting
        for the file every time."""
        return self.root / "assets"

    @property
    def sweeps(self) -> Path:
        """Sweep grids and animations. Separate from captures: a sweep is a
        study of one parameter, not a picture you chose to keep."""
        return self.root / "sweeps"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def bends(self) -> Path:
        return self.root / "library" / "bends"

    @property
    def recipes(self) -> Path:
        return self.root / "library" / "recipes"

    @property
    def presets(self) -> Path:
        return self.root / "library" / "presets"

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "source": str(self.source),
            "datasets": str(self.datasets),
            "runs": str(self.runs),
            "models": str(self.models),
            "captures": str(self.captures),
            "assets": str(self.assets),
            "sweeps": str(self.sweeps),
            "library": str(self.root / "library"),
            # Outside the workspace, but Kiln's discovery scans them and users do
            # put models there, so the UI can name and open them.
            "models_pretrained": str(_PROJECT_ROOT / "models" / "pretrained"),
            "models_fine_tuned": str(_PROJECT_ROOT / "models" / "fine_tuned"),
        }


# Singleton, created on import.
workspace = Workspace()


def _nvidia_gpu_present() -> bool:
    """Is there an NVIDIA GPU on this machine, regardless of what torch thinks?

    Lets us tell "no GPU" apart from "GPU is there, but this torch build cannot
    use it" — by far the more common and more confusing case.
    """
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=4,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def get_device_info() -> dict:
    """Report GPU availability without importing torch until needed."""
    info = {"torch": False, "cuda": False, "device": "cpu", "gpus": []}
    try:
        import torch  # noqa: PLC0415

        info["torch"] = True
        info["torch_version"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda  # None for a CPU-only wheel
        if not torch.cuda.is_available():
            if not _nvidia_gpu_present():
                info["reason"] = "no_gpu"
                info["hint"] = "No NVIDIA GPU detected. Training needs CUDA; the rest of Kiln runs on CPU."
            else:
                # Only here is the (slower) driver query worth making. It turns
                # "some CUDA problem" into which wheel to install, which matters
                # most when the wheel is *newer* than the driver: that case used
                # to read as though Kiln itself wanted that CUDA version.
                import sys  # noqa: PLC0415

                from utils import cuda as cuda_pick  # noqa: PLC0415

                driver = cuda_pick.driver_version()
                info["driver"] = driver
                report = cuda_pick.diagnose(
                    torch.__version__, torch.version.cuda, driver,
                    python=sys.executable, gpu_present=True,
                )
                info["reason"] = report["reason"]
                info["hint"] = report["hint"]
        if torch.cuda.is_available():
            info["cuda"] = True
            info["device"] = "cuda"
            gpus = []
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory
                reserved = torch.cuda.memory_reserved(i)
                # driver-level free VRAM (accounts for *other* processes too)
                try:
                    with torch.cuda.device(i):
                        free_b, total_b = torch.cuda.mem_get_info()
                    total = total_b
                    free_mb = round(free_b / (1024 * 1024))
                except Exception:  # noqa: BLE001
                    free_mb = round((total - reserved) / (1024 * 1024))
                gpus.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_mem_mb": round(total / (1024 * 1024)),
                    "used_mem_mb": round(total / (1024 * 1024)) - free_mb,
                    "free_mem_mb": free_mb,
                    "used_pct": round(100 * (1 - free_mb / max(round(total / (1024 * 1024)), 1)), 1),
                })
            info["gpus"] = gpus
    except ImportError:
        info["reason"] = "no_torch"
        info["hint"] = "PyTorch is not installed in the interpreter running Kiln."
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    return info
