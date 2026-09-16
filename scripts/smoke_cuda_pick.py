"""Smoke test for picking a PyTorch build from the NVIDIA driver (offline).

The bug this guards: PyPI's default torch wheel is built for the newest CUDA
there is, so on an older driver a plain install leaves a torch that cannot see
the GPU -- and Kiln used to report that as "built for CUDA 13.0 ... check the
driver version", which reads as though Kiln required CUDA 13.

    python scripts/smoke_cuda_pick.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import cuda  # noqa: E402

# (driver, expected index name) -- the vendor's 535 is the case that started this
CASES = [
    (592.0, "cu130"),
    (580.0, "cu130"),
    (579.9, "cu129"),
    (575.51, "cu129"),
    (571.0, "cu128"),
    (570.0, "cu128"),
    (569.9, "cu126"),
    (535.23, "cu126"),      # the vendor's driver
    (525.60, "cu126"),
    (525.0, "cu118"),
    (450.80, "cu118"),
    (390.0, None),          # older than every current build
]


def check_table():
    for driver, want in CASES:
        _url, _cuda, name = cuda.build_for(driver)
        assert name == want, (driver, name, want)
    # no driver reading at all: fall back to the build that runs on 525 and up
    _url, _cuda, name = cuda.build_for(None)
    assert name == "cu126", name
    assert cuda.build_for(535.23)[0].endswith("/cu126")
    print(f"driver -> build: ok ({len(CASES)} drivers, 535.23 -> cu126)")


def check_family_minimums():
    assert cuda.min_driver_for("13.0") == 580.0
    assert cuda.min_driver_for("12.6") == 525.60
    assert cuda.min_driver_for("11.8") == 450.80
    assert cuda.min_driver_for(None) is None
    assert cuda.min_driver_for("nonsense") is None
    # the vendor's machine: a CUDA 13 wheel on a 535 driver cannot run
    assert cuda.build_is_runnable("13.0", 535.23) is False
    assert cuda.build_is_runnable("12.6", 535.23) is True
    assert cuda.build_is_runnable("12.6", None) is None
    print("family minimums: ok (CUDA 13 needs 580, 12.x needs 525)")


def check_diagnosis():
    # the vendor's exact case
    d = cuda.diagnose("2.14.0+cu130", "13.0", 535.23, python="/k/.venv/bin/python")
    assert d["reason"] == "build_too_new", d
    for phrase in ("needs NVIDIA driver 580", "has 535.23", "no driver update required",
                   "cu126", "--fix-torch"):
        assert phrase in d["hint"], (phrase, d["hint"])

    cpu = cuda.diagnose("2.14.0", None, 535.23)
    assert cpu["reason"] == "cpu_build" and "CPU-only" in cpu["hint"], cpu
    assert "cu126" in cpu["hint"], cpu

    old = cuda.diagnose("2.5.1+cu121", "12.1", 390.0)
    assert old["reason"] == "driver_too_old" and "only fix" in old["hint"], old

    fine = cuda.diagnose("2.6.0+cu126", "12.6", 535.23)
    assert fine["reason"] == "driver" and "did not initialise" in fine["hint"], fine

    nogpu = cuda.diagnose("2.14.0", None, None, gpu_present=False)
    assert nogpu["reason"] == "cpu_build" and "GPU is present" not in nogpu["hint"], nogpu
    print("diagnosis: ok (too-new build, CPU wheel, ancient driver, healthy pairing)")


def check_installer_wiring():
    out = subprocess.run([sys.executable, str(ROOT / "install.py"), "--help"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert "--fix-torch" in out.stdout, out.stdout
    src = (ROOT / "install.py").read_text(encoding="utf-8")
    assert "CUDA_INDEX_URL" not in src, "install.py still hardcodes one CUDA index"
    assert "cuda_pick.build_for" in src and "_torch_constraints" in src
    print("installer wiring: ok (--fix-torch offered, no hardcoded index, torch pinned)")


def check_this_machine():
    driver = cuda.driver_version()
    if driver is None:
        print("this machine: no NVIDIA driver readable (skipped)")
        return
    _url, cuda_ver, name = cuda.build_for(driver)
    print(f"this machine: driver {driver:g} -> {name} (CUDA {cuda_ver})")


if __name__ == "__main__":
    check_table()
    check_family_minimums()
    check_diagnosis()
    check_installer_wiring()
    check_this_machine()
    print("smoke_cuda_pick: all ok")
