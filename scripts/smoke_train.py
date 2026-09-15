"""Dev smoke test for training across both backends.

Runs genuinely short runs (a handful of steps) against a throwaway workspace and
a tiny generated dataset, and checks the thing that actually matters for the UI:
that both engines produce the same *shape* of run -- a Job with losses and
checkpoints, a run.json, model-N snapshots and sample-N.png previews -- so the
Train and Models screens work without knowing which one produced it.

The xurdif leg needs CUDA (its trainer hardcodes .cuda()) and is skipped without
it. The Diffusers legs run anywhere.

    python scripts/smoke_train.py            # everything available
    python scripts/smoke_train.py --diffusers-only
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKSPACE = Path(tempfile.mkdtemp(prefix="kiln_train_ws_"))
os.environ["KILN_WORKSPACE"] = str(WORKSPACE)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app.core import backends  # noqa: E402
from app.core.engine.trainer import list_checkpoints, load_run_view  # noqa: E402

DATASET = WORKSPACE / "datasets" / "tiny"


def make_dataset(n: int = 12, size: int = 64):
    DATASET.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    for i in range(n):
        # Smooth low-frequency blobs, not noise: a model can actually reduce
        # loss on these, so a falling loss curve means the loop works.
        small = torch.rand(3, 4, 4)
        img = torch.nn.functional.interpolate(
            small[None], size=(size, size), mode="bicubic", align_corners=False)[0]
        img = img.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(img).save(DATASET / f"img{i:02d}.png")
    return DATASET


def wait(job, timeout=900):
    t0 = time.time()
    while job.status == "running" and time.time() - t0 < timeout:
        time.sleep(0.5)
    # The status flips before the worker's last act, patching run.json; join
    # the thread so the file is settled before anything reads it.
    thread = getattr(job, "thread", None)
    if thread is not None:
        thread.join(timeout=30)
    return job


def check_run_shape(out_dir: Path, job, label: str, expect_dirs: bool):
    assert job.status == "done", f"{label}: status {job.status} - {job.message}"

    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta.get("status") == "done", meta.get("status")

    ckpts = list_checkpoints(out_dir, meta.get("save_every"))
    assert ckpts, f"{label}: no checkpoints listed"
    for c in ckpts:
        p = Path(c["path"])
        assert p.exists(), c["path"]
        assert p.is_dir() == expect_dirs, f"{label}: unexpected checkpoint shape {p}"
        assert c["size_mb"] > 0, f"{label}: checkpoint reports zero size"
        assert c["step"] is not None, f"{label}: checkpoint has no step"

    view = load_run_view(out_dir)
    assert view["checkpoints"], f"{label}: run view lists no checkpoints"
    assert view["losses"], f"{label}: run view parsed no losses from the log"
    assert view["status"] == "done", view["status"]

    samples = sorted(out_dir.glob("sample-*.png"))
    assert samples, f"{label}: no snapshot previews rendered"

    print("  %-18s %d checkpoints (%s), %d loss points, %d previews, last loss %.4f"
          % (label, len(ckpts), "dirs" if expect_dirs else "files",
             len(view["losses"]), len(samples), view["losses"][-1]["loss"]))
    return ckpts


def run_diffusers(mode: str, run_name: str, base=None, **over):
    backend = backends.get("diffusers")
    out_dir = WORKSPACE / "runs" / run_name
    body = {"mode": mode, "preset": "small-64", "image_size": 64, "batch_size": 4,
            "train_steps": 20, "save_every": 10, "lr": 1e-4, "diffusion_steps": 1000,
            "model_name": run_name, "base_model": base, **over}
    cfg = backend.training_config(body, DATASET, out_dir)
    job = wait(backend.start_training(cfg))
    if job.status != "done":
        print("  log tail:", (job.detail.get("log") or [])[-6:])
    return out_dir, job


def check_loss_defaults():
    """Each engine's own default: xurdif trains with edge loss, Diffusers with MSE."""
    xcfg = backends.get("xurdif").training_config({
        "image_size": 64, "batch_size": 2, "train_steps": 100, "save_every": 50,
        "model_name": "x",
    }, DATASET, WORKSPACE / "runs" / "_defaults_x")
    assert xcfg.edge_loss is True, xcfg.edge_loss
    dcfg = backends.get("diffusers").training_config({
        "mode": "scratch", "preset": "small-64", "image_size": 64, "batch_size": 4,
        "train_steps": 20, "save_every": 10, "model_name": "d",
    }, DATASET, WORKSPACE / "runs" / "_defaults_d")
    assert dcfg.objective == "mse", dcfg.objective
    print("  loss defaults: xurdif edge on, diffusers MSE")


def check_xurdif(run_name: str = "xur", **over):
    """One short xurdif run. With no overrides this is what a fresh Train
    screen launches: the conf architecture on its default attention layout."""
    if not torch.cuda.is_available():
        print("  skipped: the xurdif trainer is CUDA-only and no GPU is visible")
        return
    backend = backends.get("xurdif")
    out_dir = WORKSPACE / "runs" / run_name
    cfg = backend.training_config({
        # xurdif's own validators floor these at 100 / 10.
        "image_size": 64, "batch_size": 2, "train_steps": 100, "save_every": 50,
        "accum": 1, "diffusion_steps": 1000, "nsamples": 1, "sample_seed": 42,
        "model_name": run_name, "mults": [1, 2, 2, 2], **over,
    }, DATASET, out_dir)
    assert cfg.image_size == 64 and cfg.save_every == 50 and cfg.mults == [1, 2, 2, 2]
    job = wait(backend.start_training(cfg), timeout=900)
    if job.status != "done":
        print("  log tail:", (job.detail.get("log") or [])[-8:])
    ckpts = check_run_shape(out_dir, job, f"xurdif scratch ({cfg.mtype})", expect_dirs=False)

    # the produced checkpoint must be a first-class model everywhere else
    b, ref = backends.resolve(ckpts[-1]["path"])
    assert b.name == "xurdif"
    meta = b.describe(ref)
    assert meta.mtype == cfg.mtype, (meta.mtype, cfg.mtype)
    assert meta.attn == cfg.attn, (meta.attn, cfg.attn)
    meta_json = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta_json.get("attn") == cfg.attn, meta_json.get("attn")
    if cfg.attn:
        # the patched Trainer.save wrote both keys, and the network rebuilds
        # from them with the attention modules where the layout says
        data = torch.load(ckpts[-1]["path"], map_location="cpu", weights_only=False)
        # recorded the way upstream records it: the --attn string plus the options
        assert data.get("attn_conf") == cfg.attn, data.keys()
        assert isinstance(getattr(data.get("opt"), "attn_config", None), dict), data.get("opt")
        net, _ = b.load(ref, device="cpu")
        assert type(net.down_attns[-1]).__name__ == "LinearAttention2d", type(net.down_attns[-1])
        ids = [n["id"] for n in b.layer_graph(net, image_size=64)["nodes"]]
        assert "down_attns.3" in ids and "mid_attn" in ids, ids
    print("  trained checkpoint loads back as:", meta.mtype, meta.mults, meta.attn)

    # and resuming the run keeps the same layout without being told it again
    from app.core.engine.trainer import config_from_run
    rcfg = config_from_run(out_dir, 200)
    assert rcfg.mtype == cfg.mtype and rcfg.attn == cfg.attn, (rcfg.mtype, rcfg.attn)
    rjob = wait(backend.start_training(rcfg), timeout=900)
    if rjob.status != "done":
        print("  log tail:", (rjob.detail.get("log") or [])[-8:])
    assert rjob.status == "done", f"xurdif resume: {rjob.message}"
    assert len(list_checkpoints(out_dir, 50)) >= 3, "resume added no snapshots"
    print("  resumed to step 200 on the same layout")


def check_diffusers_scratch():
    out_dir, job = run_diffusers("scratch", "dscratch")
    ckpts = check_run_shape(out_dir, job, "diffusers scratch", expect_dirs=True)

    last = ckpts[-1]["path"]
    b, ref = backends.resolve(last)
    assert b.name == "diffusers", b.name
    meta = b.describe(ref)
    assert meta.mtype == "diffusers:UNet2DModel"
    # a snapshot must be samplable and bendable like any other model
    net, _ = b.load(ref, device="cpu")
    assert b.layer_graph(net, image_size=64)["nodes"]
    print("  trained snapshot is loadable and bendable:", meta.mtype, meta.mults)
    return last


def check_diffusers_finetune(base: str):
    out_dir, job = run_diffusers("finetune", "dfine", base=base)
    check_run_shape(out_dir, job, "diffusers finetune", expect_dirs=True)


def check_tinyunet(objective: str):
    """The re-homed xurdif architecture, trained through the Diffusers loop."""
    import json

    out_dir, job = run_diffusers("scratch", f"dtiny_{objective}", preset="tiny-256",
                                 objective=objective, l1w=1.0, ssimw=0.0)
    ckpts = check_run_shape(out_dir, job, f"tinyunet {objective}", expect_dirs=True)

    snap = Path(ckpts[-1]["path"])
    b, ref = backends.resolve(str(snap))
    meta = b.describe(ref)
    assert meta.mtype == "diffusers:TinyUNet2DModel", meta.mtype
    assert meta.size_multiple == 16, meta.size_multiple

    # A run trained the xurdif way must carry the xurdif display convention,
    # and an MSE run must not.
    prov = snap / "kiln_provenance.json"
    display = json.loads(prov.read_text(encoding="utf-8")).get("display") if prov.exists() else None
    assert display == ("xurdif" if objective == "xurdif" else None), display

    # and it must still be bendable, on xurdif's own layer names
    net, _ = b.load(ref, device="cpu")
    ids = [n["id"] for n in b.layer_graph(net, image_size=64)["nodes"]]
    assert "mid_attn" in ids and "init_conv" in ids, ids[:5]
    print(f"    snapshot: {meta.mtype}, display={display!r}, {len(ids)} xurdif bend points")


def check_diffusers_lora(base: str):
    out_dir, job = run_diffusers("lora", "dlora", base=base, lora_r=4)
    ckpts = check_run_shape(out_dir, job, "diffusers LoRA", expect_dirs=True)

    snap = Path(ckpts[-1]["path"])
    adapter = snap / "adapter"
    assert adapter.is_dir(), "LoRA run wrote no adapter"
    assert (adapter / "adapter_config.json").exists()
    assert (adapter / "adapter_model.safetensors").exists()
    adapter_mb = sum(f.stat().st_size for f in adapter.rglob("*") if f.is_file()) / 2**20
    merged_mb = sum(f.stat().st_size for f in snap.glob("*") if f.is_file()) / 2**20
    assert adapter_mb < merged_mb, "the adapter should be smaller than the merged model"
    print("  adapter %.2f MB alongside a %.2f MB merged snapshot" % (adapter_mb, merged_mb))

    # the merged snapshot must load as an ordinary model, not a PEFT wrapper
    b, ref = backends.resolve(str(snap))
    net, _ = b.load(ref, device="cpu")
    assert type(net.wrapped).__name__ == "UNet2DModel", type(net.wrapped).__name__
    print("  merged LoRA snapshot loads as a plain UNet2DModel")


def check_refusals():
    """Capability gates, not backend name checks."""
    from app.backend.app import create_app

    c = create_app().test_client()
    r = c.post("/api/train", json={"dataset": "tiny", "run_name": "nope",
                                   "backend": "xurdif", "mode": "lora"})
    body = r.get_json()
    assert r.status_code == 400, r.status_code
    assert "lora" in (body.get("error") or "").lower(), body
    print("  xurdif refuses LoRA:", body.get("error"))

    r = c.post("/api/train", json={"dataset": "tiny", "run_name": "nope",
                                   "backend": "nosuch"})
    assert r.status_code == 400
    print("  unknown backend refused:", r.get_json().get("error"))

    backend = backends.get("diffusers")
    try:
        backend.training_config({"mode": "lora"}, DATASET, WORKSPACE / "runs" / "x")
    except Exception as e:
        print("  LoRA without a base model refused:", e)
    else:
        raise AssertionError("expected a ValidationError")


def main():
    diffusers_only = "--diffusers-only" in sys.argv
    try:
        make_dataset()
        print("workspace:", WORKSPACE)
        print("training defaults:")
        check_loss_defaults()
        print("training runs:")
        if not diffusers_only:
            check_xurdif()
            # the class every earlier model was trained with still trains
            check_xurdif("xur-old", mtype="tinyunet_with_attention3")
        base = check_diffusers_scratch()
        check_tinyunet("mse")
        check_tinyunet("xurdif")
        check_diffusers_finetune(base)
        check_diffusers_lora(base)
        print("capability gates:")
        check_refusals()
    finally:
        shutil.rmtree(WORKSPACE, ignore_errors=True)
    print("OK")


if __name__ == "__main__":
    main()
