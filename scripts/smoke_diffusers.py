"""Dev smoke test for the Diffusers backend (CPU-only, no downloads).

Builds tiny UNet2DModel repositories on disk rather than pulling from the Hub,
so the test is offline, fast and deterministic. If the reference DDPM model
happens to be in the local Hub cache it is exercised too, since that is the only
way to check a real-world legacy repo layout.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app.core import backends  # noqa: E402
from app.core.backends.hfdiffusers import loader  # noqa: E402
from app.core.craft import bending  # noqa: E402
from app.core.engine.sampler import SampleParams, sampler  # noqa: E402
from app.core.model_manager import manager  # noqa: E402

REFERENCE_REPO = "google/ddpm-ema-celebahq-256"


def _tiny_model(dest: Path, seed: int = 0):
    """A small but structurally complete UNet2DModel repo, with attention."""
    from diffusers import UNet2DModel

    torch.manual_seed(seed)
    net = UNet2DModel(
        sample_size=32, in_channels=3, out_channels=3, layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
    )
    net.save_pretrained(dest)
    (dest / "scheduler_config.json").write_text(json.dumps({
        "_class_name": "DDIMScheduler", "num_train_timesteps": 1000,
        "beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02,
        "prediction_type": "epsilon",
    }, indent=2), encoding="utf-8")
    loader.forget(str(dest))
    return dest


def _write(dest: Path, **files):
    dest.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        f = dest / name.replace("__", "/")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def check_discovery(tmp: Path):
    """Nothing is loaded blindly: every refusal names a reason."""
    b = backends.get("diffusers")

    latent = _write(tmp / "sd", **{
        "model_index.json": {"_class_name": "StableDiffusionPipeline",
                             "unet": ["diffusers", "UNet2DConditionModel"]},
        "unet__config.json": {"_class_name": "UNet2DConditionModel"},
    })
    info = loader.probe(str(latent))
    assert not info["supported"], info
    assert "StableDiffusionPipeline" in info["reason"], info["reason"]
    print("refuses latent pipelines:", info["reason"][:78])

    vpred = _write(tmp / "vpred", **{
        "config.json": {"_class_name": "UNet2DModel", "block_out_channels": [32, 64]},
        "scheduler_config.json": {"_class_name": "DDIMScheduler",
                                  "prediction_type": "v_prediction"},
    })
    info = loader.probe(str(vpred))
    assert not info["supported"] and "v_prediction" in info["reason"], info
    print("refuses v_prediction:", info["reason"][:78])

    other = _write(tmp / "vae", **{"config.json": {"_class_name": "AutoencoderKL"}})
    info = loader.probe(str(other))
    assert not info["supported"] and "AutoencoderKL" in info["reason"], info
    print("refuses other model classes:", info["reason"][:78])

    empty = tmp / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    assert not loader.is_local_model_dir(empty)
    assert not b.claims(str(empty))
    assert not loader.probe(str(empty))["supported"]
    print("an empty directory is not claimed")


def check_inference(model_dir: Path):
    b, ref = backends.resolve(str(model_dir))
    assert b.name == "diffusers", b.name
    meta = b.describe(ref)
    assert meta.mtype == "diffusers:UNet2DModel", meta.mtype
    assert meta.mults == [1, 2], meta.mults
    assert meta.size_multiple == 2, meta.size_multiple
    assert meta.pred == "eps"
    print("descriptor:", {k: meta.to_dict()[k] for k in
                          ("backend", "mtype", "mults", "size_multiple", "pred")})

    sched = b.schedule(ref, train_steps=4321)
    assert sched.num_train_timesteps == 1000, "the model's own T must win"
    assert sched.trained_betas is None and sched.beta_schedule == "linear"
    print("schedule ignores the caller's train_steps and uses the model's own T=1000")

    # display: no contrast stretch, unlike xurdif
    flat = torch.randn(1, 3, 8, 8) * 0.05
    dxf = b.to_display(flat)
    xur = backends.get("xurdif").to_display(flat)
    assert abs(float(dxf.std()) - float(flat.std() / 2)) < 0.01, float(dxf.std())
    assert abs(float(xur.std()) - 0.18) < 0.02, float(xur.std())
    print("to_display leaves contrast alone (std %.4f) where xurdif forces %.4f"
          % (dxf.std(), xur.std()))

    params = SampleParams(model_path=str(model_dir), image_size=32, steps=3,
                          sampler="unipc", seed=5, device="cpu", postproc={})
    frames = list(sampler.run(params))
    assert frames and frames[-1]["image"].size == (32, 32)
    print("sampled %d frames through Kiln's own loop -> %s"
          % (len(frames), frames[-1]["image"].size))


def check_craft(model_dir: Path):
    b, ref = backends.resolve(str(model_dir))
    net, _ = b.load(ref, device="cpu")

    graph = b.layer_graph(net, image_size=32)
    nodes = graph["nodes"]
    assert nodes, "layer graph is empty -- this was the pre-M4 failure"
    stages = {n["stage"] for n in nodes}
    assert {"encoder", "mid", "decoder"} <= stages, stages
    assert all(n["channels"] for n in nodes), "every node needs a channel count"
    print("layer graph: %d nodes, stages %s" % (len(nodes), sorted(stages)))
    print("  first:", [(n["id"], n["label"]) for n in nodes[:3]])

    # every group chip the UI offers must resolve to something
    for group in bending.GROUPS:
        hit = bending._resolve_targets([group], net, b)
        assert hit, f"group '{group}' resolved to nothing"
    print("all %d group chips resolve" % len(bending.GROUPS))

    # the ids the graph reports are the ids bending can hook
    ids = [n["id"] for n in nodes]
    assert bending._resolve_targets(ids, net, b) == set(ids)

    # and a bend actually changes the output
    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([10], dtype=torch.long)
    with torch.no_grad():
        plain = net(x, t).clone()
    rt = bending.build_runtime(
        [{"op": "multiply", "params": {"value": 3.0}, "targets": ["mid"],
          "step_start": 0.0, "step_end": 1.0, "active": True}],
        backend=b,
    )
    rt.attach(net)
    rt.set_total(1)
    rt.set_step(0)
    try:
        with torch.no_grad():
            bent = net(x, t).clone()
    finally:
        rt.detach()
    delta = float((bent - plain).abs().mean())
    assert delta > 1e-6, "bend had no effect on the output"
    with torch.no_grad():
        restored = net(x, t)
    assert torch.equal(restored, plain), "hooks leaked after detach"
    print("bending the mid stage changes the output (mean |delta| %.5f) and detaches cleanly"
          % delta)


def check_merge(a: Path, bdir: Path, out: Path):
    from app.core.craft.merging import check_compat, merge

    compat = check_compat(str(a), str(bdir))
    assert compat["compatible"], compat["reasons"]
    print("two identical-config Diffusers models are mergeable")

    backend, ref_a = backends.resolve(str(a))
    _, ref_b = backends.resolve(str(bdir))
    sd_a = backend.merge_slots(ref_a)["unet"]
    sd_b = backend.merge_slots(ref_b)["unet"]
    assert not torch.equal(sd_a["conv_in.weight"], sd_b["conv_in.weight"]), \
        "fixture models must actually differ"

    # alpha 0 must reproduce A exactly, alpha 1 must reproduce B exactly --
    # the check that catches a wrong stage map in blockwise mode.
    for method, alpha, blocks, want, label in (
        ("linear", 0.0, {}, sd_a, "linear a=0 -> A"),
        ("linear", 1.0, {}, sd_b, "linear a=1 -> B"),
        ("blockwise", 0.0, {"encoder": 0.0, "mid": 0.0, "decoder": 0.0}, sd_a,
         "blockwise all-0 -> A"),
        ("blockwise", 0.0, {"encoder": 1.0, "mid": 1.0, "decoder": 1.0}, sd_b,
         "blockwise all-1 -> B"),
    ):
        res = merge(str(a), str(bdir), "m", method=method, alpha=alpha,
                    block_weights=blocks, out_dir=str(out))
        got = backend.merge_slots(backends.parse_ref(res["path"]))["unet"]
        bad = [k for k in want if not torch.allclose(got[k], want[k], atol=1e-6)]
        assert not bad, f"{label}: {len(bad)}/{len(want)} tensors differ, e.g. {bad[:3]}"
        print("  %-24s exact across %d tensors" % (label, len(want)))
        shutil.rmtree(res["path"], ignore_errors=True)
        loader.forget(res["path"])

    # a merged model must be a real, rediscoverable Diffusers repo
    res = merge(str(a), str(bdir), "blend", method="slerp", alpha=0.5, out_dir=str(out))
    assert loader.is_local_model_dir(Path(res["path"])), res["path"]
    found = backend.scan([(out, "workspace")])
    assert any(m.name == "blend" for m in found), [m.name for m in found]
    print("a merged model is written as a loadable repo and rediscovered by scan")


def check_cross_backend(diff_dir: Path, tmp: Path):
    """Two engines' checkpoints are never mergeable; say so plainly."""
    from app.core.craft.merging import check_compat
    from app.core.engine.arch import build_unet

    pt = tmp / "xur.pt"
    unet = build_unet("tinyunet_with_attention3", [1, 2, 2, 2])
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    torch.save({"step": 0, "model": state, "ema": state, "mults": [1, 2, 2, 2],
                "mtype": "tinyunet_with_attention3", "pred": "x0"}, pt)

    compat = check_compat(str(pt), str(diff_dir))
    assert not compat["compatible"]
    assert any("backend" in r for r in compat["reasons"]), compat["reasons"]
    print("cross-backend merge refused:", compat["reasons"][0])


def check_reference_repo():
    """Exercise the legacy flat layout, if it is already cached locally."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        info = loader.probe(REFERENCE_REPO)
    except Exception as e:  # noqa: BLE001
        print("reference repo not checked (%s)" % e)
        return
    if not info.get("supported"):
        print("reference repo not in the local cache; skipped")
        return
    assert info["pipeline"] == "DDPMPipeline"
    assert info["unet_subfolder"] is None, "flat layout puts the UNet at the root"
    meta = loader.describe(REFERENCE_REPO)
    assert meta.size_multiple == 32, meta.size_multiple
    assert meta.mults == [1, 1, 2, 2, 4, 4], meta.mults
    print("legacy flat repo layout handled: %s, mults %s, size_multiple %d"
          % (info["pipeline"], meta.mults, meta.size_multiple))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="kiln_diffusers_"))
    try:
        check_discovery(tmp)
        a = _tiny_model(tmp / "models" / "tiny_a", seed=1)
        b = _tiny_model(tmp / "models" / "tiny_b", seed=2)
        check_inference(a)
        check_craft(a)
        check_merge(a, b, tmp / "out")
        check_cross_backend(a, tmp)
        check_reference_repo()
    finally:
        manager.clear_cache()
        loader.forget()
        shutil.rmtree(tmp, ignore_errors=True)
    print("OK")


if __name__ == "__main__":
    main()
