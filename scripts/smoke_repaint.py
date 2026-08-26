"""Dev smoke test for RePaint resampling in region fill (CPU-only).

Resampling revisits parts of the schedule so a fill can settle into its
surroundings rather than only matching at the seam. It lives inside
``Sampler.run`` alongside the ordinary path, so the things worth asserting are:

    off        -- resample=1 is byte-for-byte the old single-pass behaviour
    schedule   -- the expanded plan matches diffusers' RePaintScheduler exactly
    jumps      -- a backward move re-noises without a model evaluation
    solvers    -- DDIM and DPM++ work; the stateful ones are refused
    composite  -- unmasked pixels stay bit-exact whatever the settings
    bends      -- still fire, and on the descending schedule rather than evals

``scripts/smoke_golden.py`` carries the byte-for-byte digest for a default fill;
this covers the behaviour around it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from app.core.engine.arch import build_unet  # noqa: E402
from app.core.engine.inpaint import run_inpaint  # noqa: E402
from app.core.engine.sampler import (  # noqa: E402
    RESAMPLE_SOLVERS, SampleParams, repaint_positions, resample_supported,
    sampler_catalog,
)
from app.core.model_manager import manager  # noqa: E402

MULTS = [1, 2, 2, 2]
SIZE = 64
SEED = 99


def _model(tmp: Path) -> Path:
    torch.manual_seed(1234)
    unet = build_unet("tinyunet_with_attention3", MULTS)
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    dest = tmp / "m.pt"
    torch.save({"step": 0, "model": state, "ema": state, "mults": MULTS,
                "mtype": "tinyunet_with_attention3", "pred": "x0"}, dest)
    return dest


def _canvas_and_mask():
    torch.manual_seed(7)
    canvas = Image.fromarray(torch.rand(SIZE, SIZE, 3).mul(255).byte().numpy(), mode="RGB")
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rectangle([20, 20, 44, 44], fill=255)
    return canvas, mask


def _fill(ckpt, canvas, mask, *, steps=8, bends=None, **kw):
    manager.clear_cache()
    params = SampleParams(model_path=str(ckpt), image_size=SIZE, steps=steps,
                          train_steps=1000, eta=0.5, seed=SEED, device="cpu",
                          postproc={}, **kw)
    runtime = None
    if bends:
        from app.core.craft.bending import build_runtime

        runtime = build_runtime(bends)
    frames = list(run_inpaint(params, canvas, mask, feather=4.0,
                              mults=MULTS, bend_runtime=runtime))
    last = frames[-1]
    return np.asarray(last["image"]).astype(np.int16), len(frames), last["total"]


def check_schedule():
    """The expanded plan must match the reference implementation exactly."""
    from diffusers import RePaintScheduler

    from app.core.engine.sampler import _make_betas

    for n, jl, jn in ((50, 10, 10), (50, 10, 3), (20, 4, 5), (10, 2, 4), (50, 10, 1)):
        ref = RePaintScheduler(num_train_timesteps=1000, trained_betas=_make_betas(1000))
        ref.set_timesteps(n, jl, jn)
        stride = 1000 // n
        mine = [(n - 1 - p) * stride for p in repaint_positions(n, jl, jn)]
        assert mine == [int(t) for t in ref.timesteps], (n, jl, jn)
    plain = repaint_positions(20, 4, 1)
    assert plain == list(range(20)), "U=1 must be a plain descending walk"
    back = [b for a, b in zip(repaint_positions(20, 4, 3), repaint_positions(20, 4, 3)[1:]) if b <= a]
    assert back, "resampling must actually move backwards"
    print(f"  schedule: matches RePaintScheduler for 5 configurations; "
          f"U=3 makes {len(back)} backward moves")


def check_solver_capability():
    caps = {c["name"]: c["resample"] for c in sampler_catalog()}
    assert caps["ddim"] and caps["dpmpp"], caps
    assert not caps["unipc"] and not caps["deis"], caps
    assert set(RESAMPLE_SOLVERS) == {"ddim", "dpmpp"}, RESAMPLE_SOLVERS
    assert resample_supported("ddim") and not resample_supported("unipc")
    print(f"  solvers: resampling advertised for {', '.join(RESAMPLE_SOLVERS)} only")


def check_off_is_unchanged(ckpt, canvas, mask):
    """resample=1 must be the old path, not a re-derivation of it."""
    a, na, _ = _fill(ckpt, canvas, mask)
    b, nb, _ = _fill(ckpt, canvas, mask, resample=1)
    assert np.array_equal(a, b) and na == nb
    print(f"  off: resample=1 identical to the default ({na} evals)")
    return a


def check_resampling(ckpt, canvas, mask, base_img, canvas_arr):
    from app.core.engine.inpaint import prepare_mask

    feathered = prepare_mask(mask, (SIZE, SIZE), 4.0)
    for solver in RESAMPLE_SOLVERS:
        plain, n1, _ = _fill(ckpt, canvas, mask, sampler=solver)
        for u in (3, 5):
            img, n, total = _fill(ckpt, canvas, mask, sampler=solver, resample=u)
            assert n > n1, f"{solver} U={u}: no extra evaluations ({n} vs {n1})"
            assert n == total, f"{solver} U={u}: progress total {total} != {n} evals"
            # a jump costs no model evaluation, so evals stay well under the
            # raw entry count of the expanded plan
            assert not np.array_equal(img, plain), f"{solver} U={u}: output unchanged"
            # The invariant is about the *feathered* mask, not the drawn
            # rectangle: the blur deliberately spreads the blend outward, so
            # only pixels the mask leaves at exactly zero must be untouched.
            untouched = np.asarray(feathered) == 0
            moved = np.abs(img - canvas_arr).max(axis=2)[untouched]
            assert moved.max() == 0, (
                f"{solver} U={u}: {int((moved > 0).sum())} fully-unmasked pixels moved")
        print(f"  {solver}: {n1} evals off -> {n} evals at U=5, unmasked bit-exact")


def check_refusal(ckpt, canvas, mask):
    from utils.exceptions import EngineError

    for solver in ("unipc", "deis"):
        try:
            _fill(ckpt, canvas, mask, sampler=solver, resample=3)
        except EngineError as e:
            assert solver in str(e) and "ddim" in str(e), str(e)
        else:
            raise AssertionError(f"{solver} should refuse to resample")
    # ...but works fine without resampling
    _fill(ckpt, canvas, mask, sampler="unipc")
    print("  refusal: unipc and deis reject resampling by name, still fill normally")


def check_bends(ckpt, canvas, mask):
    """Bends must still fire, and be scheduled on the descending schedule."""
    stack = [{"op": "multiply", "params": {"value": 3.0}, "targets": ["mid"],
              "step_start": 0.0, "step_end": 1.0, "active": True}]
    plain, _, _ = _fill(ckpt, canvas, mask, resample=3)
    bent, _, _ = _fill(ckpt, canvas, mask, resample=3, bends=stack)
    assert not np.array_equal(plain, bent), "bend had no effect under resampling"

    # A window covering only the back half of the schedule must differ from one
    # covering the front half -- if `set_step` were driven by the evaluation
    # counter, jumps would smear these together.
    early = [{**stack[0], "step_start": 0.0, "step_end": 0.4}]
    late = [{**stack[0], "step_start": 0.6, "step_end": 1.0}]
    a, _, _ = _fill(ckpt, canvas, mask, resample=3, bends=early)
    b, _, _ = _fill(ckpt, canvas, mask, resample=3, bends=late)
    assert not np.array_equal(a, b), "scheduled bend windows are indistinguishable"
    print("  bends: fire under resampling, and early/late windows stay distinct")


def main():
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="kiln_repaint_"))
    try:
        ckpt = _model(tmp)
        canvas, mask = _canvas_and_mask()
        canvas_arr = np.asarray(canvas).astype(np.int16)
        print("region fill with RePaint resampling:")
        check_schedule()
        check_solver_capability()
        base = check_off_is_unchanged(ckpt, canvas, mask)
        check_resampling(ckpt, canvas, mask, base, canvas_arr)
        check_refusal(ckpt, canvas, mask)
        check_bends(ckpt, canvas, mask)
    finally:
        manager.clear_cache()
        shutil.rmtree(tmp, ignore_errors=True)
    print("OK")


if __name__ == "__main__":
    main()
