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
    check_interleaved_repro(out)

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

    def counting(self, x_s, *a, with_pp=True, **kw):
        calls.append((x_s.shape[0], with_pp))
        return real(self, x_s, *a, with_pp=with_pp, **kw)

    sampler_mod.Sampler._to_images = counting
    try:
        params = SampleParams(model_path=str(ckpt), image_size=64, steps=6, device="cpu",
                              batch_size=2, seed=7, postproc={"contrast": 1.2})
        last = None
        for frame in sampler.run(params):
            assert frame["step"] >= 1 and frame["batch"] == 2   # free to read
            last = frame
        assert not calls, f"frames rendered before anyone read them: {len(calls)}"
        # A live preview: the first variation, raw, with no post-processing.
        last["image"]
        assert calls == [(1, False)], f"a raw preview rendered {calls}"
        pp = last["image_pp"]
        assert calls[-1] == (1, True) and len(calls) == 2
        # The preview renders the first variation alone; it has to be the same
        # picture the full batch render gives for that item.
        assert len(last["images_pp"]) == 2 and calls[-1] == (2, True)
        assert last["images_pp"][0].tobytes() == pp.tobytes(), "preview differs from item 0"
        n = len(calls)
        last["images"], last["images_pp"], last["image"], last["image_pp"]
        assert len(calls) == n, "a rendered frame rendered again"
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
    print("lazy frames: no render until read, raw item-0 preview skips postproc, "
          "map_images deferred")


def check_interleaved_repro(ckpt: Path):
    """A run's image does not depend on what else is sampling at the same time.

    Two runs stepped alternately in one thread interleave their random draws
    exactly as two concurrent runs do: DDIM's eta noise, extra noise, and the
    guidance cutouts. Each has to come out byte-identical to its solo render,
    which is the promise every capture's recipe makes.
    """
    def params(seed, text=""):
        return SampleParams(model_path=str(ckpt), image_size=64, steps=4, device="cpu",
                            seed=seed, eta=0.5, noise_level=0.9, sampler="ddim",
                            text=text, guidance_step=0.05, postproc={})

    def alone(p):
        last = None
        for frame in sampler.run(p):
            last = frame
        return last["image"].tobytes()

    guided, plain = params(5, "a red bird on a branch"), params(9)
    want = [alone(guided), alone(plain)]

    runs = [sampler.run(guided), sampler.run(plain)]
    last = [None, None]
    live = [True, True]
    while any(live):
        for i, run in enumerate(runs):
            if live[i]:
                try:
                    last[i] = next(run)
                except StopIteration:
                    live[i] = False
    got = [f["image"].tobytes() for f in last]
    assert got[0] == want[0], "a guided run changed because another run sampled alongside it"
    assert got[1] == want[1], "a plain run changed because another run sampled alongside it"
    print("interleaved runs reproduce their solo renders (eta, extra noise, guidance)")


if __name__ == "__main__":
    main()
