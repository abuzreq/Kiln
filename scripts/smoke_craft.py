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

    out.unlink(missing_ok=True)
    (ROOT / "workspace_smoke_craft_conf.pt").unlink(missing_ok=True)
    print("OK")


if __name__ == "__main__":
    main()
