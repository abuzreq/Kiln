"""Dev smoke test for the crafting layer (introspection, ops, bend runtime)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app.core.engine.arch import build_unet  # noqa: E402
from app.core.model_manager import manager  # noqa: E402
from app.core.craft.introspect import introspect  # noqa: E402
from app.core.craft import ops, bending  # noqa: E402
from app.core.engine.sampler import sampler, SampleParams  # noqa: E402


def main():
    mtype, mults = "tinyunet_with_attention3", [1, 2, 2, 2]
    unet = build_unet(mtype, mults)
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    ckpt = {"step": 0, "model": state, "ema": state, "mults": mults, "mtype": mtype, "pred": "x0"}
    out = ROOT / "workspace_smoke_craft.pt"
    torch.save(ckpt, out)

    bundle = manager.load(str(out), device="cpu")
    graph = introspect(bundle["model"])
    print(f"introspected {len(graph['nodes'])} bend points; stages:",
          sorted({n['stage'] for n in graph['nodes']}))
    print("first nodes:", [(n['id'], n['type'], n['channels'], f"{n['h']}x{n['w']}") for n in graph['nodes'][:4]])
    print(f"op catalog: {len(ops.catalog())} ops")

    # The configurable-attention model adds encoder attention points; they must
    # sit in the encoder stage and answer to the "attention" group chip.
    from app.core.backends.xurdif import attn as attn_spec
    conf_spec = "-2:window,-1:linear,mid:full"
    conf = build_unet("tinyunet_conf_attention", mults, attn_config=attn_spec.parse(conf_spec))
    cstate = {f"denoise_fn.{k}": v for k, v in conf.state_dict().items()}
    cout = ROOT / "workspace_smoke_craft_conf.pt"
    import argparse
    torch.save({"step": 0, "model": cstate, "ema": cstate, "mults": mults,
                "mtype": "tinyunet_conf_attention", "pred": "x0", "attn_conf": conf_spec,
                "opt": argparse.Namespace(model="tinyunet_conf_attention", mults=mults, pred="x0",
                                          attn=conf_spec, attn_config=attn_spec.parse(conf_spec))},
               cout)
    cbundle = manager.load(str(cout), device="cpu")
    cgraph = introspect(cbundle["model"])
    by_id = {n["id"]: n for n in cgraph["nodes"]}
    assert "down_attns.2" in by_id and "down_attns.3" in by_id and "down_attns.0" not in by_id, list(by_id)
    assert by_id["down_attns.3"]["stage"] == "encoder" and by_id["down_attns.3"]["type"] == "attention"
    assert by_id["down_attns.3"]["attn_kind"] == "linear" and by_id["down_attns.2"]["attn_kind"] == "window"
    assert by_id["down_attns.3"]["label"] == "enc 3 · attention", by_id["down_attns.3"]["label"]
    order = cgraph["order"]
    assert order.index("downs.3.0") < order.index("down_attns.3") < order.index("downs.3.1"), order
    from app.core import backends as _backends
    xb = _backends.get("xurdif")
    chip = bending._resolve_targets(["attention"], cbundle["model"], xb)
    assert chip == {"down_attns.2", "down_attns.3", "mid_attn"}, chip
    enc = bending._resolve_targets(["encoder"], cbundle["model"], xb)
    assert {"down_attns.2", "down_attns.3"} <= enc, enc
    print(f"conf model: {len(cgraph['nodes'])} bend points incl. encoder attention at levels 2 and 3")

    # The scheduled window must actually open. Use a DEFAULT-shaped window
    # (spatial ops start at 0.15) rather than forcing 0..1 — the removed preview
    # passed 0..1 explicitly, which is exactly why it never caught that its
    # runtime sat at frac=0 and skipped 8 of 17 ops.
    import torch as _torch
    from app.core.craft.introspect import capture_activation

    model = bundle["model"]
    probe = _torch.randn(1, 3, 64, 64)
    t = _torch.tensor([10.0])

    windowed = [{"op": "multiply", "params": {"value": 4.0}, "targets": ["mid_block1"],
                 "step_start": 0.15, "step_end": 0.85, "active": True}]
    clean = capture_activation(model, "mid_block1", probe, t)

    rt = bending.build_runtime(windowed)
    rt.set_total(10)
    rt.set_step(0)                      # frac 0.0 -> before the window
    outside = capture_activation(model, "mid_block1", probe, t, rt)

    rt2 = bending.build_runtime(windowed)
    rt2.set_total(10)
    rt2.set_step(5)                     # frac ~0.56 -> inside the window
    inside = capture_activation(model, "mid_block1", probe, t, rt2)

    assert _torch.allclose(outside, clean), "bend fired outside its scheduled window"
    assert not _torch.allclose(inside, clean), "bend did NOT fire inside its window"
    ratio = (inside.std() / clean.std()).item()
    print(f"schedule honoured: silent at frac 0, active at frac 0.56 (std x{ratio:.2f})")
    assert clean.shape == inside.shape, "capture_activation changed the tensor shape"
    print(f"capture_activation returns {tuple(clean.shape)} for mid_block1")

    # sample with a bend runtime attached
    rt = bending.build_runtime([{"op": "add", "params": {"value": 0.3}, "targets": ["encoder"],
                                 "step_start": 0, "step_end": 1, "active": True}])
    params = SampleParams(model_path=str(out), image_size=64, steps=3, device="cpu")
    frames = list(sampler.run(params, bend_runtime=rt))
    print(f"bended sampling produced {len(frames)} frames; last size {frames[-1]['image'].size}")

    check_bend_isolation(out)
    check_discovery_baseline()
    check_slice_targets(unet, conf)

    out.unlink(missing_ok=True)
    (ROOT / "workspace_smoke_craft_conf.pt").unlink(missing_ok=True)
    print("OK")


def check_discovery_baseline():
    """Every discovery carries the unbent render to compare it against.

    The drawer shows the pair without sampling anything, so the path has to come
    back with the entry. Archives written before the picture was kept have the
    embedding but no file, and must report that honestly rather than hand the UI
    a path to nothing.
    """
    import json
    import tempfile
    from PIL import Image

    from app.core.craft.explore import Archive, BASELINE_NAME

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "arch"
        d.mkdir()
        (d / "index.json").write_text(json.dumps({
            "version": 1, "model_path": "nonexistent.pt", "metric": "clip",
            "threshold": 0.1, "tried": 1, "accepted": 1, "baseline": [0.1, 0.2],
            "entries": [{"id": "aa11", "bends": [], "novelty": 0.5,
                         "created_at": 0.0, "embedding": [0.1, 0.2]}],
        }), encoding="utf-8")
        Image.new("RGB", (8, 8)).save(d / "aa11.png")

        a = Archive.from_dir(d)
        assert a is not None
        # An embedding on its own is not a picture: an old archive says so.
        assert a.baseline_image() is None
        assert a.public_entries()[0]["baseline"] is None, "promised a file that is not there"

        Image.new("RGB", (8, 8)).save(d / BASELINE_NAME)
        a = Archive.from_dir(d)
        got = a.public_entries()[0]["baseline"]
        assert got == str(d / BASELINE_NAME), got
        assert Path(got).exists()
        # One render per archive, shared: every entry points at the same file.
        assert len({e["baseline"] for e in a.public_entries()}) == 1
    print("  discoveries carry the unbent render, and admit when they have none")


def check_bend_isolation(ckpt: Path):
    """A bend changes only the run that applied it.

    Bend hooks are registered on the model's modules, and every run on that
    model shares one cached module. A bent run and a plain run on the same
    model, stepped alternately in one thread, each have to come out exactly as
    they do alone -- without isolation the plain run is bent from the moment
    the other attaches its hooks.
    """
    stack = [{"op": "add", "params": {"value": 0.5}, "targets": ["encoder"],
              "step_start": 0, "step_end": 1, "active": True}]

    def params(seed):
        return SampleParams(model_path=str(ckpt), image_size=64, steps=4, device="cpu",
                            seed=seed, sampler="ddim", eta=0.0, postproc={})

    def alone(p, rt=None):
        last = None
        for frame in sampler.run(p, bend_runtime=rt):
            last = frame
        return last["image"].tobytes()

    plain_alone = alone(params(3))
    bent_alone = alone(params(4), bending.build_runtime(stack))
    assert alone(params(3), bending.build_runtime(stack)) != plain_alone, "the bend did nothing"

    runs = [sampler.run(params(3)), sampler.run(params(4), bend_runtime=bending.build_runtime(stack))]
    last = [None, None]
    live = [True, True]
    while any(live):
        for i, run in enumerate(runs):
            if live[i]:
                try:
                    last[i] = next(run)
                except StopIteration:
                    live[i] = False
    assert last[0]["image"].tobytes() == plain_alone, "a plain run was bent by another run's stack"
    assert last[1]["image"].tobytes() == bent_alone, "a bent run changed when run alongside another"
    print("bends stay in their own run: plain and bent runs on one model, interleaved, match solo")


def check_slice_targets(unet, conf):
    """Skips and q/k/v bend only their own channels.

    A skip target rewrites the skip half of a decoder block's input and
    nothing else: the upsampled half arrives untouched, and the result differs
    from bending the skip's source block (which also feeds the path down). A
    q/k/v part rewrites its third of the fused projection's output.
    """
    from app.core import backends as _backends
    from app.core.backends.xurdif import graph as xgraph
    from app.core.craft.interchange import to_external

    xb = _backends.get("xurdif")
    torch.manual_seed(0)
    x, t = torch.randn(1, 3, 64, 64), torch.tensor([10.0])

    g = xb.layer_graph(unet)
    skips = {s["id"]: s for s in g["skips"]}
    assert set(skips) == {f"skip:{i}" for i in range(len(unet.downs))}, list(skips)
    assert skips["skip:1"]["from"] == "downs.1.0" and skips["skip:1"]["to"] == f"ups.{len(unet.downs) - 2}.1"
    inner = {n["id"]: n for n in g["inner"]}
    assert {"downs.1.0.conv", "downs.1.0.norm", "downs.1.0.film"} <= set(inner), sorted(inner)[:6]
    assert "downs.1.0.act" not in inner, "act duplicates the block's own output"
    assert {"mid_attn.q", "mid_attn.k", "mid_attn.v", "mid_attn.proj"} <= set(inner)
    assert inner["downs.1.0.film"]["parent"] == "downs.1.0" and inner["downs.1.0.film"]["stage"] == "encoder"
    # Groups keep meaning the main points: no stack changes what it bends.
    enc = bending._resolve_targets(["encoder"], unet, xb)
    assert not any(":" in n or n in inner for n in enc), enc

    def run(model, targets, op="multiply", params=None, probe=None):
        """Output for one bend on `targets`, plus what `probe` module saw going in."""
        seen = {}
        hooks = []
        rt = bending.build_runtime([{"op": op, "params": params or {"value": 0.0}, "targets": targets,
                                     "step_start": 0, "step_end": 1, "active": True}], backend=xb)
        rt.attach(model)
        if probe:
            # registered after the bend, so it sees what the block really gets
            def look(m, a):
                seen.setdefault("in", a[0].clone())     # returns None: a probe, not a bend
            hooks.append(dict(model.named_modules())[probe].register_forward_pre_hook(look))
        try:
            with torch.no_grad():
                out = model(x, t)
        finally:
            rt.detach()
            for h in hooks:
                h.remove()
        return out, seen.get("in")

    with torch.no_grad():
        plain = unet(x, t)
    consumer = skips["skip:1"]["to"]
    split = xgraph.skip_points(unet)[1]["start"]
    same, _ = run(unet, ["skip:1"], params={"value": 1.0})
    assert torch.equal(same, plain), "multiply 1 on a skip changed the output"
    zeroed, seen = run(unet, ["skip:1"], probe=consumer)
    _, seen_plain = run(unet, [], probe=consumer)
    assert not torch.equal(zeroed, plain), "multiply 0 on a skip did nothing"
    assert torch.equal(seen[:, :split], seen_plain[:, :split]), "the upsampled half was touched"
    assert torch.count_nonzero(seen[:, split:]) == 0, "the skip half was not bent"
    source, _ = run(unet, [skips["skip:1"]["from"]])
    assert not torch.equal(zeroed, source), "bending the skip is the same as bending its source"

    cg = xb.layer_graph(conf)
    parts = {n["id"] for n in cg["inner"]}
    assert {"mid_attn.qkv:q", "mid_attn.qkv:k", "mid_attn.qkv:v", "mid_attn.proj"} <= parts, sorted(parts)
    assert "down_attns.3.to_qkv:v" in parts and "down_attns.3.to_out" in parts, sorted(parts)
    with torch.no_grad():
        cplain = conf(x, t)
    seen_out = {}
    qkv = dict(conf.named_modules())["mid_attn.qkv"]
    h = qkv.register_forward_hook(lambda m, i, o: seen_out.setdefault("o", o.clone()))
    vzero, _ = run(conf, ["mid_attn.qkv:v"])
    h.remove()
    c = qkv.out_channels // 3
    assert not torch.equal(vzero, cplain), "zeroing v did nothing"
    # the hook above ran before the bend's, so compare the bent output instead
    seen_bent = {}
    rt = bending.build_runtime([{"op": "multiply", "params": {"value": 0.0}, "targets": ["mid_attn.qkv:v"],
                                 "step_start": 0, "step_end": 1, "active": True}], backend=xb)
    rt.attach(conf)
    h = qkv.register_forward_hook(lambda m, i, o: seen_bent.setdefault("o", o.clone()))
    with torch.no_grad():
        conf(x, t)
    h.remove()
    rt.detach()
    assert torch.equal(seen_bent["o"][:, :2 * c], seen_out["o"][:, :2 * c]), "q or k was touched"
    assert torch.count_nonzero(seen_bent["o"][:, 2 * c:]) == 0, "v was not bent"

    doc, report = to_external([{"op": "multiply", "params": {"value": 0.5}, "active": True,
                                "targets": ["skip:1", "downs.1.0.film", "mid_attn.qkv:v"]}])
    assert [e["path"] for e in doc["bends"]] == ["downs.1.0.film"], doc["bends"]
    assert report["kiln_only_targets"] == ["mid_attn.qkv:v", "skip:1"], report
    print(f"  skips and q/k/v bend only their channels; {len(g['inner'])} inner points, "
          f"{len(g['skips'])} skips; export leaves them out")


if __name__ == "__main__":
    main()
