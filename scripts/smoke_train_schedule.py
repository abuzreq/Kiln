"""The Diffusers trainer must noise on the schedule it saves.

A snapshot's ``scheduler_config.json`` is what every later sample -- including
the training preview -- reads its noise levels from. If the training loop builds
its scheduler from anything else, the model learns one set of noise levels and
is sampled at another. That never raises; it just produces worse images (see
``BetaSchedule`` in app/core/backends/base.py), so it has to be checked here.

For each case this builds the trainer's own noise scheduler, writes a real
snapshot, reads it back through the backend exactly as the sampler does, and
demands identical ``alphas_cumprod``:

    scratch   -- tiny-256 (cosine, pinned as trained_betas) and standard-128
                 (named linear) as the control
    finetune  -- continuing a trained_betas model must keep its curve
    lora      -- likewise (skipped without peft)

CPU only, no training steps: the schedule is decided before the first batch.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app.core import backends  # noqa: E402
from app.core.backends.hfdiffusers import loader, training  # noqa: E402

FAILURES: list[str] = []


def sampling_alphas(model_dir: Path):
    """``alphas_cumprod`` as the sampler would build it for this snapshot."""
    from diffusers import DDPMScheduler

    loader.forget(str(model_dir))
    backend, ref = backends.resolve(str(model_dir))
    assert backend.name == "diffusers", backend.name
    return DDPMScheduler(**backend.schedule(ref).scheduler_kwargs()).alphas_cumprod


def compare(label: str, trained, sampled) -> bool:
    diff = (trained - sampled).abs().max().item()
    mid = len(trained) // 2
    same = torch.equal(trained, sampled)
    print(f"  {label}: {'match' if same else 'MISMATCH'}  "
          f"max|d alphas_cumprod| {diff:.4f}  "
          f"t={mid}: trained {trained[mid]:.4f} vs sampled {sampled[mid]:.4f}")
    if not same:
        FAILURES.append(label)
    return same


def check_scratch(preset: str, tmp: Path) -> Path:
    out = tmp / preset
    cfg = training.config_from_body({"preset": preset, "image_size": 64}, str(tmp), str(out))
    net, sched_cfg, _, _ = training._build_model(cfg)
    trained = training._noise_scheduler(sched_cfg).alphas_cumprod
    dest = training._save_snapshot(net, cfg, sched_cfg, out, 1)
    compare(f"scratch {preset}", trained, sampling_alphas(dest))
    return dest


def check_continue(mode: str, base: Path, tmp: Path):
    import json

    out = tmp / f"{mode}-of-{base.parent.name}"
    # The base is a TinyUNet, whose modules are not named like UNet2DModel's, so
    # the default LORA_TARGETS would not attach; these are its equivalents.
    cfg = training.config_from_body(
        {"mode": mode, "base_model": str(base), "image_size": 64, "lora_r": 4,
         "lora_targets": ["conv", "q", "k", "v", "proj"]},
        str(tmp), str(out))
    net, sched_cfg, _, _ = training._build_model(cfg)
    base_cfg = json.loads((base / "scheduler_config.json").read_text(encoding="utf-8"))
    kept = sched_cfg.get("trained_betas") == base_cfg.get("trained_betas")
    print(f"  {mode}: trained_betas {'kept' if kept else 'DROPPED'} from the base")
    if not kept:
        FAILURES.append(f"{mode} trained_betas")

    base_alphas = sampling_alphas(base)
    compare(f"{mode} training vs base", training._noise_scheduler(sched_cfg).alphas_cumprod,
            base_alphas)
    dest = training._save_snapshot(net, cfg, sched_cfg, out, 1)
    compare(f"{mode} snapshot vs base", sampling_alphas(dest), base_alphas)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="kiln_sched_"))
    try:
        print("trainer noise schedule vs saved schedule:")
        tiny = check_scratch("tiny-256", tmp)
        check_scratch("standard-128", tmp)
        check_continue("finetune", tiny, tmp)
        try:
            import peft  # noqa: F401
        except ImportError:
            print("  lora: peft not installed, skipped")
        else:
            check_continue("lora", tiny, tmp)
    finally:
        loader.forget()
        shutil.rmtree(tmp, ignore_errors=True)
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
