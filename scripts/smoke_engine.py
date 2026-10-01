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

    check_lazy_frames(out)

    out.unlink(missing_ok=True)
    print("OK")


def check_lazy_frames(ckpt: Path):
    """Frames render only when read, once, and a deferred transform rides along.

    The sampler used to render every step whether or not anyone looked; callers
    that read only the last frame paid for all of them. Counting calls to
    ``_to_images`` is the direct test that they no longer do.
    """
    from app.core.engine import sampler as sampler_mod

    calls = []
    real = sampler_mod.Sampler._to_images

    def counting(self, *a, **kw):
        calls.append(1)
        return real(self, *a, **kw)

    sampler_mod.Sampler._to_images = counting
    try:
        params = SampleParams(model_path=str(ckpt), image_size=64, steps=6, device="cpu",
                              batch_size=2, seed=7)
        last = None
        for frame in sampler.run(params):
            assert frame["step"] >= 1 and frame["batch"] == 2   # free to read
            last = frame
        assert not calls, f"frames rendered before anyone read them: {len(calls)}"
        img = last["image_pp"]
        last["image"]
        assert len(calls) == 1, f"the preview item rendered {len(calls)} times"
        # The preview renders the first variation alone; it has to be the same
        # picture the full batch render gives for that item.
        assert len(last["images"]) == 2 and len(calls) == 2
        assert last["images_pp"][0].tobytes() == img.tobytes(), "preview differs from item 0"
        last["images_pp"], last["image"]
        assert len(calls) == 2, "a rendered frame rendered again"
        assert set(last) == {"step", "total", "batch", "image", "image_pp",
                             "images", "images_pp"}

        # map_images is deferred too, and applies to every image the frame yields.
        seen = []
        frame = next(iter(sampler.run(SampleParams(
            model_path=str(ckpt), image_size=64, steps=2, device="cpu", seed=7))))
        frame.map_images(lambda im: seen.append(im) or im.transpose(0))
        assert not seen, "map_images ran before the frame was read"
        frame["image_pp"]
        assert len(seen) == 2 and "images" not in frame
    finally:
        sampler_mod.Sampler._to_images = real
    print("lazy frames: no render until read, preview renders item 0 only, "
          "map_images deferred")


if __name__ == "__main__":
    main()
