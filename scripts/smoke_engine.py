"""Dev smoke test for the engine layer (runs on CPU, no training required).

Builds an untrained UNet, saves it as a xurdif-format checkpoint, then loads it
through the model manager and runs a few DDIM steps through the sampler.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app.core.engine.arch import build_unet, available_architectures  # noqa: E402
from app.core.model_manager import read_meta, manager  # noqa: E402
from app.core.engine.sampler import sampler, SampleParams  # noqa: E402


def main():
    print("available architectures:", available_architectures())
    mtype, mults = "tinyunet_with_attention3", [1, 2, 2, 2]
    unet = build_unet(mtype, mults)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"built {mtype} with {n_params/1e6:.2f}M params")

    # wrap into a xurdif-style checkpoint (GaussianDiffusion prefixes 'denoise_fn.')
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    ckpt = {"step": 0, "model": state, "ema": state, "mults": mults, "mtype": mtype, "pred": "x0"}
    out = ROOT / "workspace_smoke_model.pt"
    torch.save(ckpt, out)

    meta = read_meta(out)
    print("meta:", meta.to_dict())

    params = SampleParams(model_path=str(out), image_size=64, steps=4, device="cpu")
    frames = list(sampler.run(params))
    print(f"sampled {len(frames)} frames; last image size = {frames[-1]['image'].size}")

    out.unlink(missing_ok=True)
    print("OK")


if __name__ == "__main__":
    main()
