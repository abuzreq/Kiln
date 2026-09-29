"""Differential test: the re-homed TinyUNet must equal the vendored one.

``app/core/backends/hfdiffusers/tinyunet.py`` is a deliberate op-for-op copy of
``vendor/xurdif/alt_models/tinyunet_with_attn3.py``, and
``.../objectives.py`` is the same for the vendored training loss. Copies rot.
This is the guard: it holds the two implementations side by side and asserts
they agree, at every level that could hide a difference.

    structure  -- same state dict keys, module order and class names
    outputs    -- bit-identical forwards
    gradients  -- bit-identical grads through a fixed loss
    training   -- identical loss and weights over repeated optimiser steps
    objective  -- the ported loss equals the vendored one, term for term
    algorithm  -- both networks inside the vendor's own GaussianDiffusion
    conversion -- a real .pt converts and samples to the same image

If this fails, the copy is wrong, not the test.

The GaussianDiffusion leg needs CUDA (the vendored class hardcodes ``.cuda()``)
and is skipped without it. Everything else runs on CPU.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.core import backends  # noqa: E402
from app.core.backends.hfdiffusers import objectives  # noqa: E402
from app.core.backends.hfdiffusers.convert import convert_checkpoint  # noqa: E402
from app.core.backends.hfdiffusers.tinyunet import TinyUNet2DModel  # noqa: E402
from app.core.engine._vendor import ensure_on_path  # noqa: E402
from app.core.engine.arch import build_unet  # noqa: E402
from app.core.engine.sampler import SampleParams, sampler  # noqa: E402
from app.core.model_manager import manager  # noqa: E402

ensure_on_path()
import xurdif as V  # noqa: E402

SHAPES = ([1, 2, 2, 2], [1, 2, 4], [1, 1, 2, 2, 4, 4])


def _pair(mults, seed=0):
    """A vendored net and a re-homed one holding the same weights."""
    torch.manual_seed(seed)
    ref = build_unet("tinyunet_with_attention3", list(mults))
    new = TinyUNet2DModel.from_vendored(ref)
    return ref.eval(), new.eval()


def check_structure():
    for mults in SHAPES:
        ref, new = _pair(mults)
        assert list(ref.state_dict()) == list(new.state_dict()), mults
        nm_r = [n for n, _ in ref.named_modules()]
        nm_n = [n for n, _ in new.named_modules()]
        assert nm_r == nm_n, mults
        # Root class differs by design; every inner class name must not, because
        # Craft groups layers by type(module).__name__.
        t_r = [type(m).__name__ for n, m in ref.named_modules() if n]
        t_n = [type(m).__name__ for n, m in new.named_modules() if n]
        assert t_r == t_n, mults
        assert sum(p.numel() for p in ref.parameters()) == \
               sum(p.numel() for p in new.parameters())
    types = {type(m).__name__ for _, m in new.named_modules()}
    assert {"ConvBlock", "SelfAttention2d"} <= types, types
    print("  structure: keys, module order, class names and param counts match "
          f"for {len(SHAPES)} shapes")


def check_outputs():
    for mults in SHAPES:
        ref, new = _pair(mults)
        step = 2 ** len(mults)
        for bs, size in ((1, step), (3, step * 2)):
            x = torch.randn(bs, 3, size, size)
            t = torch.randint(0, 1000, (bs,))
            with torch.no_grad():
                a = ref(x, t)
                b = new(x, t, return_dict=False)[0]
                c = new(x, t).sample
            assert torch.equal(a, b), f"{mults} bs={bs} size={size}"
            assert torch.equal(a, c), "return_dict path differs"
        # float timesteps: the layer probe passes them
        x = torch.randn(1, 3, step, step)
        with torch.no_grad():
            assert torch.equal(ref(x, torch.tensor([10.0])),
                               new(x, torch.tensor([10.0]), return_dict=False)[0])
    print("  outputs: bit-identical across shapes, batch sizes, both return paths")


def check_gradients():
    ref, new = _pair([1, 2, 2, 2], seed=5)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    target = torch.randn(2, 3, 32, 32)
    ra = ref(x, t)
    rb = new(x, t, return_dict=False)[0]
    assert torch.equal(ra, rb)
    torch.nn.functional.mse_loss(ra, target).backward()
    torch.nn.functional.mse_loss(rb, target).backward()
    ga = dict(ref.named_parameters())
    gb = dict(new.named_parameters())
    bad = [k for k in ga if not torch.equal(ga[k].grad, gb[k].grad)]
    assert not bad, f"{len(bad)}/{len(ga)} gradients differ, e.g. {bad[:3]}"
    print(f"  gradients: bit-identical across all {len(ga)} tensors")


def check_training_steps(steps=6):
    """Identical data, identical updates -- weights must not drift apart."""
    ref, new = _pair([1, 2, 2, 2], seed=11)
    ref.train(); new.train()
    oa = torch.optim.AdamW(ref.parameters(), lr=1e-3)
    ob = torch.optim.AdamW(new.parameters(), lr=1e-3)
    losses = []
    for i in range(steps):
        torch.manual_seed(100 + i)
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        target = torch.randn(2, 3, 32, 32)
        la = torch.nn.functional.mse_loss(ref(x, t), target)
        lb = torch.nn.functional.mse_loss(new(x, t, return_dict=False)[0], target)
        assert torch.equal(la, lb), f"loss diverged at step {i}: {la.item()} vs {lb.item()}"
        oa.zero_grad(); la.backward(); oa.step()
        ob.zero_grad(); lb.backward(); ob.step()
        sa, sb = ref.state_dict(), new.state_dict()
        bad = [k for k in sa if not torch.equal(sa[k], sb[k])]
        assert not bad, f"weights diverged at step {i}: {len(bad)} tensors"
        losses.append(la.item())
    assert losses[-1] < losses[0], "the fixture should actually be learning"
    print(f"  training: {steps} optimiser steps, weights identical throughout "
          f"(loss {losses[0]:.4f} -> {losses[-1]:.4f})")


def check_objective():
    """The ported loss must equal the vendored one, not merely approximate it."""
    from pytorch_msssim import ssim

    torch.manual_seed(0)
    for shape in ((2, 3, 32, 32), (1, 3, 64, 64)):
        pred = torch.randn(*shape).clamp(-1, 1)
        tgt = torch.randn(*shape).clamp(-1, 1)
        assert torch.equal(V.sobel_edges(tgt), objectives.sobel_edges(tgt))
        va, _ = V.edge_weighted_l1(x_pred=pred, x_target=tgt,
                                   edge_weight=4.0, edge_threshold=0.08)
        ma, _ = objectives.edge_weighted_l1(x_pred=pred, x_target=tgt,
                                            edge_weight=4.0, edge_threshold=0.08)
        assert torch.equal(va, ma), "edge_weighted_l1 differs"
        for l1w, ssimw in ((1.0, 0.0), (1.0, 10.0), (0.5, 2.0)):
            p_ = pred.clamp(-1, 1); t_ = tgt.clamp(-1, 1).detach()
            vl, _ = V.edge_weighted_l1(x_pred=p_, x_target=t_,
                                       edge_weight=4.0, edge_threshold=0.08)
            vs = 1.0 - ssim((p_ * 0.5 + 0.5).clamp(0, 1), t_ * 0.5 + 0.5)
            vendored = l1w * vl + ssimw * vs
            mine, _ = objectives.xurdif_objective(pred, tgt, l1w=l1w, ssimw=ssimw)
            assert torch.equal(vendored, mine), f"objective differs at {l1w}/{ssimw}"
    print("  objective: sobel, edge-weighted L1 and the composed loss all identical")


def check_edge_switch():
    """The KILN edge-loss switch: on reproduces upstream, off is a plain L1.

    ``use_edges`` defaults to True precisely so every model trained before the
    patch keeps its meaning, so the on-path is asserted equal to the hardcoded
    call it replaced -- not merely close to it.
    """
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    tgt = torch.randn(2, 3, 32, 32).clamp(-1, 1).detach()

    on, _ = V.edge_weighted_l1(x_pred=pred, x_target=tgt,
                               edge_weight=4.0, edge_threshold=0.08)
    off = (pred - tgt).abs().mean()
    assert not torch.equal(on, off), "edge weighting made no difference to the loss"

    net = build_unet("tinyunet_with_attention3", [1, 2, 2, 2])
    gd_on = V.GaussianDiffusion(net, image_size=32, timesteps=1000, l1w=1.0,
                                ssimw=0.0, pred="x0")
    gd_off = V.GaussianDiffusion(net, image_size=32, timesteps=1000, l1w=1.0,
                                 ssimw=0.0, pred="x0", use_edges=False)
    assert gd_on.use_edges and gd_on.edge_weight == 4.0 and gd_on.edge_threshold == 0.08,         "the default must still be the values the upstream call site hardcoded"
    assert not gd_off.use_edges

    x = torch.randn(1, 3, 32, 32).clamp(-1, 1)
    t = torch.full((1,), 40, dtype=torch.long)
    noise = torch.randn_like(x)
    with torch.no_grad():
        l_on = gd_on.p_losses(x, t, noise=noise)
        l_off = gd_off.p_losses(x, t, noise=noise)
    assert not torch.equal(l_on, l_off), "--noEdges changed nothing in p_losses"
    print(f"  edge switch: default matches the hardcoded call; off differs "
          f"({float(l_on):.4f} vs {float(l_off):.4f})")


def check_vendor_harness():
    """Both networks inside the vendored GaussianDiffusion: same loss, same grads.

    Gradients are compared against the hardware's own run-to-run variation
    rather than demanded exactly equal, because the vendored eps path is not
    deterministic on CUDA: running the *same* model twice differs on ~38 of 72
    tensors by up to ~1e-6, which is the ``extract()`` gather backward using
    atomics. Requiring bit-equality there would test the GPU, not the port. The
    x0 path -- which is Kiln's default -- has no such floor and does come out
    exactly equal.
    """
    if not torch.cuda.is_available():
        print("  algorithm: skipped (GaussianDiffusion hardcodes .cuda())")
        return

    class Call:
        def __init__(self, m): self.m = m
        def __call__(self, x, t): return self.m(x, t, return_dict=False)[0]

    torch.manual_seed(0)
    ref = build_unet("tinyunet_with_attention3", [1, 2, 2, 2]).cuda()
    new = TinyUNet2DModel.from_vendored(ref).cuda()

    def run(denoise_fn, owner, pred, l1w, ssimw):
        gd = V.GaussianDiffusion(denoise_fn, image_size=64, timesteps=1000,
                                 l1w=l1w, ssimw=ssimw, pred=pred).cuda()
        torch.manual_seed(7)
        x0 = torch.randn(2, 3, 64, 64, device="cuda").clamp(-1, 1)
        t = torch.randint(0, 1000, (2,), device="cuda").long()
        noise = torch.randn_like(x0)
        owner.zero_grad(set_to_none=True)
        loss = gd.p_losses(x0, t, noise=noise)
        loss = loss[0] if isinstance(loss, tuple) else loss
        loss.backward()
        return loss.detach().clone(), {k: p.grad.clone() for k, p in owner.named_parameters()}

    def worst(a, b):
        return max((a[k] - b[k]).abs().max().item() for k in a)

    for pred, l1w, ssimw in (("x0", 1.0, 0.0), ("x0", 1.0, 10.0), ("eps", 1.0, 0.0)):
        # The floor: the vendored model against itself, several times. One
        # sample is not enough -- the nondeterminism is intermittent, and a
        # lucky pair reads 1e-11 where the next reads 1e-6.
        floor = 0.0
        for _ in range(3):
            _, f1 = run(ref, ref, pred, l1w, ssimw)
            _, f2 = run(ref, ref, pred, l1w, ssimw)
            floor = max(floor, worst(f1, f2))

        la, A = run(ref, ref, pred, l1w, ssimw)
        lb, B = run(Call(new), new, pred, l1w, ssimw)
        assert torch.equal(la, lb), f"{pred}: loss {la.item()} vs {lb.item()}"
        cross = worst(A, B)
        scale = max(v.abs().max().item() for v in A.values())
        # A real porting error shows up orders of magnitude above this; the
        # tolerance only absorbs float32 atomics ordering.
        tol = max(floor, 1e-5 * scale)
        assert cross <= tol, (
            f"{pred}: implementations differ by {cross:.2e}, above the "
            f"tolerance {tol:.2e} (self-variation {floor:.2e}, "
            f"gradient scale {scale:.3f})")
        how = "exact" if cross == 0 else (
            f"{cross:.1e} vs self-variation {floor:.1e}, scale {scale:.2f}")
        print(f"    pred={pred:<3} ssimw={ssimw:<4.0f} loss identical, grads match ({how})")
    print("  algorithm: identical loss and gradients inside the vendored "
          "GaussianDiffusion")


def check_step_hook():
    """The vendored per-step hook: inert by default, effective when installed.

    The seam Kiln adds to ``Trainer`` for learning-rate scheduling is two lines,
    and its whole contract is that ``kiln_step_hook is None`` trains exactly as
    upstream did. That is what this asserts -- and then that an installed hook
    really is called, without which the first half would pass on a hook that
    never runs.
    """
    if not torch.cuda.is_available():
        print("  step hook: skipped (the vendored Trainer hardcodes .cuda())")
        return
    from types import SimpleNamespace

    tmp = Path(tempfile.mkdtemp(prefix="kiln_hook_"))

    def build(where):
        torch.manual_seed(0)
        net = build_unet("tinyunet_with_attention3", [1, 2, 2, 2]).cuda()
        gd = V.GaussianDiffusion(net, image_size=64, timesteps=1000,
                                 l1w=1.0, ssimw=0.0, pred="x0").cuda()
        torch.manual_seed(5)
        ds = [torch.randn(3, 64, 64) for _ in range(8)]
        opts = SimpleNamespace(mults=[1, 2, 2, 2], model="tinyunet_with_attention3",
                               pred="x0", attn=None, sampleSeed=-1)
        where.mkdir(parents=True, exist_ok=True)
        return V.Trainer(gd, "", image_size=64, train_batch_size=2, train_lr=1e-4,
                         train_num_steps=4, gradient_accumulate_every=1,
                         save_and_sample_every=10 ** 9, results_folder=str(where),
                         nsamples=1, opts=opts, ddim_steps=0, dataset=ds)

    # the default is the invariant
    assert build(tmp / "a").kiln_step_hook is None, "the step hook must default to None"

    def train(name, hook):
        tr = build(tmp / name)
        tr.kiln_step_hook = hook
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        tr.train()
        return tr, {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}

    _, plain = train("plain", None)
    _, noop = train("noop", lambda tr: None)
    off = [k for k in plain if not torch.equal(plain[k], noop[k])]
    assert not off, f"a no-op hook changed training: {len(off)} tensors differ, e.g. {off[0]}"

    seen = []

    def lower(tr):
        seen.append(tr.step)
        for g in tr.opt.param_groups:
            g["lr"] = 1e-6

    tr, slowed = train("slow", lower)
    assert seen == [0, 1, 2, 3], f"the hook was not called once per step: {seen}"
    assert tr.opt.param_groups[0]["lr"] == 1e-6, tr.opt.param_groups[0]["lr"]
    moved = [k for k in plain if not torch.equal(plain[k], slowed[k])]
    assert moved, "dropping the rate to 1e-6 changed nothing -- the hook cannot be working"
    shutil.rmtree(tmp, ignore_errors=True)
    print("  step hook: None trains byte-for-byte as upstream; an installed hook "
          "runs every step (%d tensors moved at lr 1e-6)" % len(moved))


def check_schedule_provenance():
    """The two vendored cosine schedules agree, and conversion pins the curve."""
    from app.core.backends.xurdif import cosine_betas

    for T in (1000, 500, 250):
        a = V.cosine_beta_schedule(T)
        a = a if torch.is_tensor(a) else torch.tensor(a)
        assert torch.equal(a.float(), torch.tensor(cosine_betas(T))), T
    # and it is NOT diffusers' named cosine, which is why conversion writes betas
    from diffusers import DDIMScheduler
    named = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2").betas
    ours = torch.tensor(cosine_betas(1000))
    assert not torch.allclose(named, ours), \
        "if these now match, conversion could record the schedule by name"
    print("  schedule: xurdif's training and sampling curves agree; diffusers' "
          f"named cosine differs by up to {(named - ours).abs().max():.3f}")


def check_conversion(tmp: Path):
    """A real checkpoint converts and samples to the same image, bit for bit."""
    for pred in ("x0", "eps"):
        torch.manual_seed(3)
        net = build_unet("tinyunet_with_attention3", [1, 2, 2, 2])
        state = {f"denoise_fn.{k}": v for k, v in net.state_dict().items()}
        pt = tmp / f"src_{pred}.pt"
        torch.save({"step": 4200, "model": state, "ema": state, "mults": [1, 2, 2, 2],
                    "mtype": "tinyunet_with_attention3", "pred": pred}, pt)
        rep = convert_checkpoint(pt, tmp / f"conv_{pred}", ema=True)
        assert rep["source_step"] == 4200 and rep["display"] == "xurdif"

        b, ref = backends.resolve(str(tmp / f"conv_{pred}"))
        meta = b.describe(ref)
        assert meta.backend == "diffusers"
        assert meta.mtype == "diffusers:TinyUNet2DModel", meta.mtype
        assert meta.mults == [1, 2, 2, 2] and meta.size_multiple == 16
        assert meta.pred == pred and meta.step == 4200

        # schedule must equal the xurdif backend's, exactly
        sx = backends.get("xurdif").schedule(None, 1000)
        sd = b.schedule(ref, 1000)
        assert np.array_equal(np.asarray(sx.trained_betas, dtype=np.float32),
                              np.asarray(sd.trained_betas, dtype=np.float32))
        assert sd.prediction_type == "epsilon", \
            "the loop hands the scheduler eps whatever the model predicts"

        for solver in ("ddim", "unipc"):
            imgs = []
            for path in (str(pt), f"diffusers:{tmp / f'conv_{pred}'}"):
                manager.clear_cache()
                p = SampleParams(model_path=path, image_size=64, steps=8,
                                 train_steps=1000, eta=0.5, seed=99,
                                 sampler=solver, device="cpu", postproc={})
                last = None
                for frame in sampler.run(p):
                    last = frame
                imgs.append(np.asarray(last["image"]).astype(np.int16))
            assert np.array_equal(*imgs), (
                f"{pred}/{solver}: converted model sampled differently "
                f"(max diff {np.abs(imgs[0] - imgs[1]).max()})")
    print("  conversion: .pt -> Diffusers repo samples bit-identically "
          "(x0 and eps, DDIM and UniPC)")


def check_craft_and_peft(tmp: Path):
    """The re-homed model must stay first-class in Craft, and gain LoRA."""
    from app.core.craft import bending

    b, ref = backends.resolve(str(tmp / "conv_x0"))
    net, meta = b.load(ref, device="cpu")
    graph = b.layer_graph(net, image_size=64)
    xg = backends.get("xurdif").layer_graph(
        build_unet("tinyunet_with_attention3", [1, 2, 2, 2]), image_size=64)
    assert [n["id"] for n in graph["nodes"]] == [n["id"] for n in xg["nodes"]], \
        "the re-homed model must expose the same bend points"
    for group in bending.GROUPS:
        assert bending._resolve_targets([group], net.wrapped, b), f"group '{group}' empty"
    # merge bucketing must use xurdif's vocabulary, not UNet2DModel's
    stages = {b.stage_of_key(k) for k in net.wrapped.state_dict()}
    assert {"encoder", "mid", "decoder"} <= stages, stages
    print(f"  craft: same {len(graph['nodes'])} bend points as the vendored model; "
          "all group chips resolve; merge stages bucket correctly")

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("  peft: not installed, skipped")
        return
    import copy

    base = copy.deepcopy(net.wrapped)
    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([10])
    with torch.no_grad():
        before = base(x, t, return_dict=False)[0].clone()
    lora = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, init_lora_weights="gaussian",
        target_modules=["conv", "q", "k", "v", "proj"]))
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    with torch.no_grad():
        after = lora(x, t, return_dict=False)[0]
    assert torch.allclose(before, after, atol=1e-6), \
        "a freshly initialised adapter must be a no-op"
    merged = lora.merge_and_unload()
    assert type(merged).__name__ == "TinyUNet2DModel", type(merged).__name__
    print(f"  peft: LoRA attaches ({trainable / 1e3:.1f}k trainable params), "
          "is a no-op at init, and merges back to a plain model")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="kiln_parity_"))
    try:
        print("re-homed TinyUNet vs vendored TinyUNetWithAttn:")
        check_structure()
        check_outputs()
        check_gradients()
        check_training_steps()
        check_objective()
        check_edge_switch()
        check_vendor_harness()
        check_step_hook()
        check_schedule_provenance()
        check_conversion(tmp)
        check_craft_and_peft(tmp)
    finally:
        manager.clear_cache()
        shutil.rmtree(tmp, ignore_errors=True)
    print("OK")


if __name__ == "__main__":
    main()
