"""xurdif golden-output regression test.

The hard requirement for the Diffusers backend work is that xurdif behaviour does
not change. This builds a deterministic checkpoint from a fixed seed, samples it
with fixed settings on the CPU, and hashes the resulting PNG bytes.

Run it before and after every milestone: the digest must not move.

    python scripts/smoke_golden.py            # print the digest
    python scripts/smoke_golden.py --check    # compare against the recorded one
"""
import hashlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

# Recorded on the pre-refactor tree. Update ONLY with a deliberate, explained
# change to xurdif behaviour -- never to make a red test go green.
GOLDEN = {
    "sample": "f179b2f53cb8244a0e923ba5ef6aae97702f8b1869abcb88ac3066016f68ab2b",
    # Region fill with resampling off. Guards the masked path the same way
    # "sample" guards plain generation -- without it, moving the fill loop is
    # unverifiable.
    "fill": "58eda00b508e761944ab26265ca698df1f7f798c97589a85d854e7743de35956",
}

MTYPE = "tinyunet_with_attention3"
MULTS = [1, 2, 2, 2]
CKPT_SEED = 1234
SAMPLE_SEED = 99


def _deterministic_checkpoint(dest: Path):
    """A xurdif-format checkpoint whose weights depend only on CKPT_SEED."""
    from app.core.engine.arch import build_unet

    torch.manual_seed(CKPT_SEED)
    unet = build_unet(MTYPE, MULTS)
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    torch.save(
        {"step": 0, "model": state, "ema": state,
         "mults": MULTS, "mtype": MTYPE, "pred": "x0"},
        dest,
    )


def _digest_sample(ckpt: Path) -> str:
    from app.core.engine.sampler import SampleParams, sampler

    params = SampleParams(
        model_path=str(ckpt), image_size=64, steps=6, train_steps=1000,
        eta=0.5, seed=SAMPLE_SEED, sampler="ddim", device="cpu", postproc={},
    )
    last = None
    for frame in sampler.run(params):
        last = frame
    buf = io.BytesIO()
    last["image"].save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _digest_fill(ckpt: Path) -> str:
    """A masked region fill, resampling off -- today's default behaviour."""
    from PIL import Image, ImageDraw

    from app.core.engine.inpaint import run_inpaint
    from app.core.engine.sampler import SampleParams

    torch.manual_seed(CKPT_SEED)
    canvas = Image.fromarray(
        (torch.rand(64, 64, 3).mul(255)).byte().numpy(), mode="RGB")
    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).rectangle([20, 20, 44, 44], fill=255)

    params = SampleParams(
        model_path=str(ckpt), image_size=64, steps=6, train_steps=1000,
        eta=0.5, seed=SAMPLE_SEED, sampler="ddim", device="cpu", postproc={},
    )
    last = None
    for frame in run_inpaint(params, canvas, mask, feather=4.0, mults=MULTS):
        last = frame
    buf = io.BytesIO()
    last["image"].save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main() -> int:
    tmp = ROOT / "_golden_model.pt"
    try:
        _deterministic_checkpoint(tmp)
        digests = {"sample": _digest_sample(tmp), "fill": _digest_fill(tmp)}
    finally:
        tmp.unlink(missing_ok=True)

    check = "--check" in sys.argv
    failed = False
    for key, got in digests.items():
        want = GOLDEN.get(key)
        if check and want != got:
            print(f"FAIL {key}: expected {want}\n           got {got}")
            failed = True
        else:
            print(f"{key}: {got}")
    if check and not failed:
        print("OK - xurdif output unchanged")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
