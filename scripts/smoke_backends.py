"""Dev smoke test for the backend seam (CPU-only, no training).

Guards the invariants that let a second engine coexist with xurdif:
identifiers round-trip unchanged, the registry dispatches correctly, and the
relocated xurdif constants are bit-identical to the ones that were inlined
before the split.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.core import backends  # noqa: E402
from app.core.backends.base import BetaSchedule, ModelRef  # noqa: E402
from app.core.engine.arch import build_unet  # noqa: E402
from app.core.model_manager import manager, read_meta  # noqa: E402


def _write_ckpt(dest: Path, mults=(1, 2, 2, 2)):
    unet = build_unet("tinyunet_with_attention3", list(mults))
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    torch.save({"step": 0, "model": state, "ema": state, "mults": list(mults),
                "mtype": "tinyunet_with_attention3", "pred": "x0"}, dest)


def main():
    assert "xurdif" in backends.available(), backends.available()
    print("registered backends:", backends.available())

    ckpt = ROOT / "_backends_smoke.pt"
    _write_ckpt(ckpt)
    try:
        # --- identifiers round-trip -----------------------------------
        ref = backends.parse_ref(str(ckpt))
        assert ref.backend == "xurdif", ref
        assert str(ref) == str(ckpt), f"{str(ref)!r} != {str(ckpt)!r}"
        assert backends.parse_ref(ref) is ref, "parse must be idempotent"
        print(f"xurdif ref round-trips byte-identically: {str(ref) == str(ckpt)}")

        # A Windows drive letter must never be read as a backend selector.
        drive = backends.parse_ref(str(ckpt.resolve()))
        assert drive.backend == "xurdif" and str(drive) == str(ckpt.resolve())
        win = "C:" + chr(92) + "models" + chr(92) + "x.pt"
        assert backends._split_prefix(win) is None
        assert backends._split_prefix("C:/models/x.pt") is None
        assert backends._split_prefix("xurdif:some/model.pt") == ("xurdif", "some/model.pt")
        print("drive letters are not mistaken for backend prefixes")

        # unknown identifiers fail loudly, naming what was tried
        try:
            backends.parse_ref("definitely-not-a-model")
        except Exception as e:
            assert "tried" in str(e), e
            print("unknown identifier rejected:", e)
        else:
            raise AssertionError("expected a NotFoundError")

        # --- dispatch -------------------------------------------------
        backend, ref = backends.resolve(str(ckpt))
        assert backend.name == "xurdif"
        caps = backend.capabilities
        assert caps.inference and caps.bend and caps.merge and caps.finetune
        assert not caps.lora, "xurdif has no LoRA path; see the plan's matrix"
        assert not caps.latent, "the pixel-space loop cannot run a latent model"
        assert caps.train_devices == ("cuda",), caps.train_devices
        print("xurdif capabilities:", caps.to_dict())

        # --- schedule is bit-identical to the old inlined constant -----
        from app.core.engine.sampler import _make_betas

        for n in (50, 100, 1000, 4000):
            sched = backend.schedule(ref, n)
            assert isinstance(sched, BetaSchedule)
            assert sched.num_train_timesteps == n
            assert np.array_equal(sched.trained_betas, _make_betas(n)), n
            assert sched.prediction_type == "epsilon"
        print("cosine schedule identical to the pre-split _make_betas for 50/100/1000/4000")

        # --- descriptor keeps the old shape, plus the additive fields --
        meta = read_meta(ckpt)
        d = meta.to_dict()
        for k in ("path", "name", "mtype", "mults", "pred", "step", "size_mb",
                  "source", "thumbnail"):
            assert k in d, k
        assert d["backend"] == "xurdif"
        assert d["size_multiple"] == 16, d["size_multiple"]
        _write_ckpt(ckpt, mults=(1, 2, 2, 2, 4))
        assert read_meta(ckpt).size_multiple == 32
        _write_ckpt(ckpt)
        print("descriptor keeps every legacy key; size_multiple tracks mults")

        # --- manager still hands back the old bundle shape ------------
        bundle = manager.load(str(ckpt), device="cpu")
        assert {"model", "meta", "backend", "ref"} <= set(bundle)
        out = bundle["model"](torch.randn(1, 3, 64, 64), torch.tensor([10]))
        assert isinstance(out, torch.Tensor) and out.shape == (1, 3, 64, 64), out.shape
        print("loaded net honours the (x, t) -> Tensor contract:", tuple(out.shape))

        # display convention stays with the backend
        disp = backend.to_display(torch.randn(2, 3, 8, 8) * 0.05)
        assert disp.min() >= 0 and disp.max() <= 1
        assert abs(float(disp[0].std()) - 0.18) < 0.02, float(disp[0].std())
        print("xurdif to_display still normalises to std 0.18")

        assert manager.evict(str(ckpt)) >= 1, "evict must match the cache key"
        print("evict matches on the ref, not a raw prefix")
    finally:
        manager.clear_cache()
        ckpt.unlink(missing_ok=True)

    print("OK")


if __name__ == "__main__":
    main()
