"""Smoke test for Kiln on macOS: generation on Apple's GPU, Diffusers-only training.

Two modes:

    python scripts/smoke_mac.py            # simulated; runs on any machine
    python scripts/smoke_mac.py --real     # on an Apple Silicon Mac, on the GPU

Simulated mode patches the platform and torch's device queries to look like an
Apple Silicon Mac (then an Intel one, then Rosetta) and checks every decision
Kiln makes from them: which device it picks, what it reports, which engines it
lets train, and what it changes about a Diffusers run on MPS. No tensor ever
touches a Metal device, so it proves the plumbing, not the GPU.

Real mode does the part simulation cannot: samples a xurdif and a Diffusers
model on MPS, trains a small Diffusers model there, and generates while that
run is going.
"""
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Kiln's messages carry "▸"; a Windows console pipe is cp1252 and cannot print it.
sys.stdout.reconfigure(errors="replace")

WORKSPACE = Path(tempfile.mkdtemp(prefix="kiln_mac_ws_"))
os.environ["KILN_WORKSPACE"] = str(WORKSPACE)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app.core import backends, devices  # noqa: E402
from utils import platform_mac  # noqa: E402

DATASET = WORKSPACE / "datasets" / "tiny"


def make_dataset(n: int = 12, size: int = 64):
    DATASET.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    for i in range(n):
        arr = (torch.rand(size, size, 3, generator=g) * 255).byte().numpy()
        Image.fromarray(arr, "RGB").save(DATASET / f"img_{i:03d}.png")


# --- simulation ---------------------------------------------------------
@contextmanager
def patched(obj, **attrs):
    saved = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(obj, k, v)


@contextmanager
def simulate(kind: str):
    """``kind``: "apple" (Apple Silicon), "intel" (Intel Mac) or "rosetta"."""
    apple, rosetta = kind == "apple", kind == "rosetta"
    gib = 1024 ** 3
    with patched(platform_mac,
                 is_mac=lambda: True,
                 is_apple_silicon=lambda: apple,
                 under_rosetta=lambda: rosetta,
                 chip_name=lambda: "Apple M2 Pro" if apple else ""), \
         patched(torch.cuda, is_available=lambda: False), \
         patched(torch.backends.mps, is_available=lambda: apple), \
         patched(torch.mps,
                 recommended_max_memory=lambda: 21 * gib,
                 driver_allocated_memory=lambda: 3 * gib):
        yield


def check_device_choice():
    from app.core.engine.sampler import pick_device

    with simulate("apple"):
        assert pick_device("auto") == "mps", pick_device("auto")
        assert pick_device("cpu") == "cpu"
        os.environ["KILN_DEVICE"] = "cpu"
        try:
            assert pick_device("auto") == "cpu", "KILN_DEVICE=cpu must win"
        finally:
            del os.environ["KILN_DEVICE"]
    with simulate("intel"):
        assert pick_device("auto") == "cpu"
    print("  auto picks mps on Apple Silicon, cpu on Intel; KILN_DEVICE overrides")


def check_lanes():
    from app.core.engine import lanes

    with simulate("apple"):
        assert lanes.default_lanes() == 1
    with simulate("intel"):
        assert lanes.default_lanes() == lanes.DEFAULT_LANES
    print("  one generation lane on Apple Silicon")


def check_device_info():
    from app.core.config import get_device_info

    with simulate("apple"):
        d = get_device_info()
        assert d["mps"] is True and d["cuda"] is False and d["device"] == "mps", d
        gpu = d["gpus"][0]
        assert gpu["name"] == "Apple M2 Pro" and gpu["shared"] is True, gpu
        assert gpu["total_mem_mb"] == 21 * 1024 and gpu["used_mem_mb"] == 3 * 1024, gpu
        assert "reason" not in d, d
    print("  Apple Silicon:", gpu["name"], f"{gpu['used_mem_mb']}/{gpu['total_mem_mb']} MiB")
    for kind, reason in (("intel", "no_gpu"), ("rosetta", "rosetta")):
        with simulate(kind):
            d = get_device_info()
        assert d["mps"] is False and d["device"] == "cpu" and d["reason"] == reason, d
        assert "NVIDIA" not in d["hint"], d["hint"]
        print(f"  {kind}: {d['hint']}")


def check_training_gate():
    from app.backend.app import create_app

    c = create_app().test_client()

    # Off the Mac nothing changes: xurdif stays the default and trainable.
    d = c.get("/api/train/backends").get_json()["data"]
    if not platform_mac.is_mac():
        assert d["default"] == "xurdif", d["default"]
        assert all(e["trainable_here"] for e in d["backends"]), d

    with simulate("apple"):
        d = c.get("/api/train/backends").get_json()["data"]
        by = {e["name"]: e for e in d["backends"]}
        assert d["default"] == "diffusers", d["default"]
        assert by["diffusers"]["trainable_here"] is True
        assert by["xurdif"]["trainable_here"] is False
        assert "NVIDIA" in by["xurdif"]["unavailable_reason"]
        print("  /train/backends:", by["xurdif"]["unavailable_reason"])

        r = c.post("/api/train", json={"dataset": "tiny", "run_name": "nope",
                                       "backend": "xurdif", "mode": "scratch"})
        assert r.status_code == 400 and "NVIDIA" in r.get_json()["error"], r.get_json()
        # A capability refusal still says what is wrong with the request.
        r = c.post("/api/train", json={"dataset": "tiny", "run_name": "nope",
                                       "backend": "xurdif", "mode": "lora"})
        assert r.status_code == 400 and "lora" in r.get_json()["error"].lower()
        r = c.post("/api/runs/anyrun/continue", json={"train_steps": 1000})
        assert r.status_code == 400 and "NVIDIA" in r.get_json()["error"], r.get_json()
        print("  xurdif training and continuing refused on the Mac")

    for kind in ("intel", "rosetta"):
        with simulate(kind):
            d = c.get("/api/train/backends").get_json()["data"]
            assert not any(e["trainable_here"] for e in d["backends"]), d
            r = c.post("/api/train", json={"dataset": "tiny", "run_name": "nope",
                                           "backend": "diffusers", "mode": "scratch"})
            assert r.status_code == 400, r.get_json()
            print(f"  {kind}: {r.get_json()['error']}")


def check_mps_overrides():
    from app.core.backends.hfdiffusers.training import _device_overrides

    backend = backends.get("diffusers")
    body = {"mode": "scratch", "preset": "small-64", "image_size": 64, "batch_size": 4,
            "train_steps": 20, "save_every": 10, "precision": "fp16", "compile": True}
    cfg = backend.training_config(body, DATASET, WORKSPACE / "runs" / "o1")
    assert _device_overrides(cfg, "cuda") == []
    assert cfg.precision == "fp16" and cfg.compile_model is True
    notes = _device_overrides(cfg, "mps")
    assert len(notes) == 2 and cfg.precision == "no" and cfg.compile_model is False, notes
    print("  Diffusers on MPS: fp32, no torch.compile")


def simulated():
    print("device choice:")
    check_device_choice()
    check_lanes()
    print("device report:")
    check_device_info()
    print("training gate:")
    check_training_gate()
    check_mps_overrides()


# --- real hardware --------------------------------------------------------
def _xurdif_checkpoint(dest: Path):
    from app.core.engine.arch import build_unet

    torch.manual_seed(1234)
    unet = build_unet("tinyunet_with_attention3", [1, 2, 2, 2])
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    torch.save({"step": 0, "model": state, "ema": state, "mults": [1, 2, 2, 2],
                "mtype": "tinyunet_with_attention3", "pred": "x0"}, dest)


def _sample(model_path: str, device: str, **over):
    import numpy as np

    from app.core.engine.sampler import SampleParams, sampler

    params = SampleParams(model_path=model_path, image_size=64, steps=8, train_steps=1000,
                          seed=7, sampler="ddim", device=device, postproc={}, **over)
    last = None
    for frame in sampler.run(params):
        last = frame
    a = np.asarray(last["image"].convert("RGB"), dtype=np.float32)
    assert np.isfinite(a).all(), "non-finite pixels"
    return a


def _train_diffusers(run_name: str, steps: int = 30):
    from app.core.backends.hfdiffusers import training as dtrain

    backend = backends.get("diffusers")
    body = {"mode": "scratch", "preset": "small-64", "image_size": 64, "batch_size": 4,
            "train_steps": steps, "save_every": 10, "lr": 1e-4, "diffusion_steps": 1000,
            "precision": "fp16"}
    cfg = backend.training_config(body, DATASET, WORKSPACE / "runs" / run_name)
    return dtrain.start_training(cfg)


def _wait(job, timeout=1800):
    t0 = time.time()
    while job.status == "running" and time.time() - t0 < timeout:
        time.sleep(0.5)
    if getattr(job, "thread", None) is not None:
        job.thread.join(timeout=30)
    return job


def real():
    import math

    assert devices.best_device() == "mps", (
        f"--real needs an Apple Silicon Mac with MPS; this one gives {devices.best_device()}")
    print("sampling on MPS:")
    ckpt = WORKSPACE / "xur.pt"
    _xurdif_checkpoint(ckpt)
    for dev in ("mps", "cpu"):
        a = _sample(str(ckpt), dev)
        print(f"  xurdif on {dev}: mean {a.mean():.1f} std {a.std():.1f}")
    if "--clip" in sys.argv:
        a = _sample(str(ckpt), "mps", text="a red apple", text_weight=1.0)
        print(f"  xurdif + CLIP guidance on mps: mean {a.mean():.1f}")

    print("training a Diffusers model on MPS:")
    job = _wait(_train_diffusers("mac1"))
    log = job.detail.get("log") or []
    assert job.status == "done", f"{job.status}: {job.message}\n" + "\n".join(log[-10:])
    assert job.detail.get("device") == "mps", job.detail.get("device")
    assert any("fp32" in line for line in log), "fp16 request was not overridden"
    losses = [p["loss"] for p in job.detail.get("losses") or []]
    assert losses and all(math.isfinite(x) for x in losses), losses[-5:]
    snaps = sorted((WORKSPACE / "runs" / "mac1").glob("model-*"))
    assert snaps, "no snapshot written"
    print(f"  {len(losses)} losses, last {losses[-1]:.4f}; snapshot {snaps[-1].name}")
    a = _sample(str(snaps[-1]), "mps")
    print(f"  the snapshot samples on mps: mean {a.mean():.1f}")

    print("generating while training (both on the GPU):")
    job = _train_diffusers("mac2", steps=60)
    errors = []

    def gen():
        try:
            while job.status == "running":
                _sample(str(ckpt), "mps")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=gen)
    t.start()
    _wait(job)
    t.join()
    assert job.status == "done" and not errors, (job.message, errors)
    print("  ok")


def main():
    try:
        make_dataset()
        if "--real" in sys.argv:
            real()
        else:
            simulated()
        print("OK")
    finally:
        import shutil

        shutil.rmtree(WORKSPACE, ignore_errors=True)


if __name__ == "__main__":
    main()
