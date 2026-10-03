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
import subprocess
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


def check_lr_plan():
    """The schedule evaluator: presets, shapes, round-trip, refusals. No GPU."""
    import math

    from app.core.engine import lr_plan as lrplan
    from utils.exceptions import ValidationError

    for lr, total in ((5e-4, 280000), (4e-4, 120000), (3e-4, 400000)):
        for pid in lrplan.PRESETS:
            plan = lrplan.compile_plan({"preset": pid}, lr=lr, train_steps=total,
                                       save_every=1000)
            assert plan["segments"][0]["from"] == 0
            assert abs(lrplan.lr_at(plan, 0) - lr) < 1e-15, (pid, lr)
            assert json.loads(json.dumps(plan)) == plan, pid
            assert lrplan.summarize(plan), pid

    # xurdif's author's own pattern: two drops, at 30% and 65% of a 280k run
    d2 = lrplan.compile_plan({"preset": "drops-2"}, lr=5e-4, train_steps=280000,
                             save_every=1000)
    assert [s["from"] for s in d2["segments"]] == [0, 84000, 182000], d2["segments"]
    assert [s["lr"] for s in d2["segments"]] == [5e-4, 1e-4, 5e-5], d2["segments"]
    # drops never go up, and every breakpoint lands on a snapshot boundary so
    # there is a checkpoint from just before each one to compare against
    for pid in ("drops-2", "drops-3"):
        plan = lrplan.compile_plan({"preset": pid}, lr=5e-4, train_steps=280000,
                                   save_every=1000)
        assert all(s["from"] % 1000 == 0 for s in plan["segments"]), plan
        seen = [lrplan.lr_at(plan, s) for s in range(0, 280001, 1000)]
        assert all(b <= a + 1e-18 for a, b in zip(seen, seen[1:])), pid

    # the smooth curve must stay the cosine it claims to be, not "some decay"
    cos = lrplan.compile_plan({"preset": "cosine-floor"}, lr=5e-4, train_steps=280000,
                              save_every=1000)
    seg = cos["segments"][0]
    hi, lo, until = seg["lr"], seg["to_lr"], seg["until"]
    for i in range(51):
        s = int(i * until / 50)
        want = lo + (hi - lo) * 0.5 * (1 + math.cos(math.pi * min(s / until, 1.0)))
        assert abs(lrplan.lr_at(cos, s) - want) < 1e-12, (s, lrplan.lr_at(cos, s), want)
    assert abs(lrplan.lr_at(cos, 10 ** 9) - lo) < 1e-18, "a ramp must hold its floor"

    # a cycle's troughs are where the good snapshots are, so they must land on
    # save_every boundaries -- which needs an even multiple of it as the period
    cyc = lrplan.compile_plan({"preset": "cyclic"}, lr=5e-4, train_steps=280000,
                              save_every=1000)
    seg = cyc["segments"][0]
    per = seg["period"]
    assert per % 2000 == 0 and (per // 2) % 1000 == 0, per
    assert abs(lrplan.lr_at(cyc, 0) - seg["lr"]) < 1e-18, "a cycle starts high"
    for k in range(3):
        assert abs(lrplan.lr_at(cyc, per // 2 + k * per) - seg["to_lr"]) < 1e-18, k
        assert abs(lrplan.lr_at(cyc, k * per) - seg["lr"]) < 1e-18, k

    # a one-click drop keeps the history in front of it
    dropped = lrplan.override_from(d2, 100000, 1e-6)
    assert abs(lrplan.lr_at(dropped, 0) - 5e-4) < 1e-18
    assert abs(lrplan.lr_at(dropped, 99999) - 1e-4) < 1e-18
    assert abs(lrplan.lr_at(dropped, 100000) - 1e-6) < 1e-18

    refusals = [
        ({"segments": []}, "no segments"),
        ({"segments": [{"from": 5, "kind": "const", "lr": 1e-4}]}, "first not at step 0"),
        ({"segments": [{"from": 0, "kind": "const", "lr": 1e-4},
                       {"from": 0, "kind": "const", "lr": 1e-5}]}, "out of order"),
        ({"segments": [{"from": 0, "kind": "const", "lr": -1e-4}]}, "negative rate"),
        ({"segments": [{"from": 0, "kind": "const", "lr": 5.0}]}, "rate above 1"),
        ({"segments": [{"from": 0, "kind": "cosine", "lr": 1e-4, "to_lr": 1e-5,
                        "until": 0}]}, "until not after from"),
        ({"segments": [{"from": 0, "kind": "cyclic", "lr": 1e-4, "to_lr": 1e-5,
                        "period": 7}]}, "odd period"),
        ({"segments": [{"from": 0, "kind": "wobble", "lr": 1e-4}]}, "unknown shape"),
        ({"preset": "nope"}, "unknown preset"),
    ]
    for spec, why in refusals:
        try:
            lrplan.compile_plan(spec, lr=1e-4, train_steps=1000, save_every=100)
        except ValidationError:
            continue
        raise AssertionError("a plan with %s should have been refused" % why)

    print("  %d schedules compile; cosine matches the closed form; %d refusals hold"
          % (len(lrplan.PRESETS), len(refusals)))


def check_lr_log_merge():
    """Relaunching a run must not throw away the loss history.

    train.log used to be truncated on every launch, which erased exactly the half
    of the curve you need in order to see whether a rate drop helped. No GPU:
    this drives the rotation and the parser directly.
    """
    from app.core.engine.trainer import _log_segments, _rotate_run_log

    out = WORKSPACE / "runs" / "_logmerge"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"name": "lm", "save_every": 50}),
                                  encoding="utf-8")

    def launch(steps, loss):
        with open(_rotate_run_log(out), "w", encoding="utf-8") as f:
            for s in steps:
                f.write("%d: %s\n" % (s, loss))

    launch(range(0, 100, 25), "0.5")             # first launch, steps 0..75
    launch(range(100, 200, 25), "0.3")           # continued on to 175
    assert (out / "train.log.1").exists(), "the first launch's log was destroyed"
    assert len(_log_segments(out)) == 2, _log_segments(out)
    steps = [p["step"] for p in load_run_view(out)["losses"]]
    assert steps == [0, 25, 50, 75, 100, 125, 150, 175], steps

    # continuing from an *earlier* checkpoint: the newer launch supersedes the
    # overlapping tail instead of drawing a zigzag back down the x axis
    launch(range(50, 150, 25), "0.1")
    pts = load_run_view(out)["losses"]
    steps = [p["step"] for p in pts]
    assert steps == sorted(steps), steps
    assert steps == [0, 25, 50, 75, 100, 125], steps
    assert all(p["loss"] == 0.1 for p in pts if p["step"] >= 50), pts
    print("  log rotation keeps %d launches; the curve spans them and stays ordered"
          % len(_log_segments(out)))


def check_vram_estimate():
    """A wider network must cost more than a narrow one, or the preset badges lie."""
    from app.core.engine.trainer import estimate_peak_mib, param_count, parse_mults

    narrow, standard, deep = [1, 2, 2, 2], [1, 2, 2, 4], [1, 2, 2, 4, 4]
    assert param_count(narrow) < param_count(standard) < param_count(deep)
    seen = [estimate_peak_mib(512, 4, m, amp=True) for m in (narrow, standard, deep)]
    assert seen[0] < seen[1] < seen[2], seen
    # the fit was measured at 1,2,2,2 with AMP, so that point must not have moved
    assert abs(seen[0] - 3760) < 20, seen[0]
    assert parse_mults("1,2,2,4") == standard and parse_mults(None) == narrow
    print("  VRAM estimate tracks width: %s MiB at 512px batch 4 for %s / %s / %s"
          % (seen, narrow, standard, deep))


def check_vram_estimate_amp():
    """Running without AMP must cost more, and the estimate has to say so.

    It did not, and the omission was expensive: the fit only ever covered the
    AMP case while TrainConfig and the form both default AMP off, so a 256px
    batch-8 run was estimated at 2327 MiB and really used about 4.2 GB. Nothing
    warned, and the run died on a 6 GB card -- silently, because CUDA falls into
    the Windows shared-memory fallback rather than raising.
    """
    from app.core.engine.trainer import estimate_peak_mib, recommended_batch

    mults = [1, 2, 2, 2]
    on = estimate_peak_mib(256, 8, mults, amp=True)
    off = estimate_peak_mib(256, 8, mults, amp=False)
    assert off > on, (on, off)
    # The default is the conservative one: guessing high only costs a warning,
    # guessing low costs the run.
    assert estimate_peak_mib(256, 8, mults) == off

    # Only activations scale with AMP, so difference two batch sizes to isolate
    # them -- context, weights, grads, moments and the EMA copy all cancel.
    def act(amp):
        return (estimate_peak_mib(512, 5, mults, amp=amp)
                - estimate_peak_mib(512, 1, mults, amp=amp)) / 4.0
    ratio = act(False) / act(True)
    assert 1.9 <= ratio <= 2.05, ratio          # measured 1.96 on an RTX 3060

    # and the batch advice has to move with it, or it recommends a batch that
    # cannot run: this card was told 23 when 11 was the honest answer.
    assert recommended_batch(256, 6144, mults, amp=False) < \
           recommended_batch(256, 6144, mults, amp=True)
    print("  AMP: %d MiB on / %d MiB off at 256px batch 8; activations x%.2f without it"
          % (on, off, ratio))


def check_amp_default():
    """A new run trains in mixed precision; a resumed one keeps what it had.

    AMP roughly halves the activation term, and without it the standard 512
    preset does not fit a 6 GB card at all. But a run that started without it
    has to carry on without it, or continuing a run silently changes its
    numerics partway through the loss curve.
    """
    from app.core.engine.trainer import TrainConfig

    assert TrainConfig(dataset="d", out_dir="o", name="n").amp is True

    b = backends.get("xurdif")
    body = {"image_size": 64, "batch_size": 1, "train_steps": 100, "save_every": 50}
    assert b.training_config(dict(body), "d", "o").amp is True
    assert b.training_config(dict(body, amp=False), "d", "o").amp is False, \
        "the Advanced checkbox must still be able to turn it off"
    print("  AMP defaults on for a new run, and stays off when asked")


def check_child_stream_decoding():
    """A tqdm progress bar must not be able to kill the log reader.

    tqdm draws with U+2588. Read back with the locale encoding -- cp1252 on a
    default Windows install -- that raised UnicodeDecodeError inside
    ``for line in proc.stdout``, which killed the reader thread. With nobody
    draining the pipe the trainer then blocked forever on write() as soon as the
    8KB buffer filled, which is the first snapshot. It looked like a hung GPU:
    the process sat at 0 CPU with the sampling bar frozen part-drawn, and the
    job stayed "running" for ever because the thread died before it could set a
    final status.

    The child writes raw UTF-8 bytes, so the case is reproduced whatever the
    child's own locale is -- what is under test here is the reader.
    """
    from app.core.engine.trainer import _CHILD_IO

    child = "\n".join([
        "import sys",
        "FULL = chr(0x2588)",
        # tqdm draws fractional progress with the partial blocks U+2589..U+258F.
        # U+258D encodes to E2 96 8D, and 0x8D is one of the five bytes cp1252
        # leaves undefined -- that is the byte that actually killed the reader.
        "PARTIAL = [chr(c) for c in range(0x2589, 0x2590)]",
        "for i in range(400):",
        "    line = chr(13) + '%3d%%|' % (i * 100 // 400) + FULL * 18 + PARTIAL[i % 7] + '| %d/400' % i",
        "    sys.stdout.buffer.write(line.encode('utf-8'))",
        "    sys.stdout.buffer.flush()",
        "sys.stdout.buffer.write(chr(10).join(['', 'done', '']).encode('utf-8'))",
    ])
    proc = subprocess.Popen([sys.executable, "-u", "-c", child], **_CHILD_IO)
    lines = [ln.rstrip() for ln in proc.stdout]        # the decode that used to raise
    proc.wait(timeout=60)
    assert proc.returncode == 0, proc.returncode
    assert any(ln.endswith("done") for ln in lines), lines[-3:]
    assert any(chr(0x2588) in ln for ln in lines), "block characters did not survive the decode"
    # Bytes, not characters: it is the encoded size that fills the OS pipe, and
    # filling it is what turned a dead reader into a deadlocked trainer.
    written = sum(len(x.encode("utf-8")) for x in lines)
    assert written > 16384, "child wrote %d bytes, too little to fill the pipe buffer" % written
    print("  log reader survives a %d-line tqdm bar (%d KiB, well past the 8 KiB pipe buffer)"
          % (len(lines), written // 1024))


def check_live_lr_edit():
    """Changing the rate mid-run, end to end on the engine that needs no GPU."""
    from app.core.engine import lr_plan as lrplan

    backend = backends.get("diffusers")
    out_dir = WORKSPACE / "runs" / "dlive"
    cfg = backend.training_config({
        "mode": "scratch", "preset": "small-64", "image_size": 64, "batch_size": 4,
        "train_steps": 100, "save_every": 50, "lr": 1e-4, "model_name": "dlive",
    }, DATASET, out_dir)
    assert cfg.lr_plan and cfg.lr_plan["preset"] == "constant", cfg.lr_plan

    job = backend.start_training(cfg)
    edited = False
    t0 = time.time()
    while job.status == "running" and time.time() - t0 < 600:
        step = (job.detail or {}).get("step") or 0
        if not edited and step >= 30:
            lrplan.write(out_dir, lrplan.override_from(lrplan.read(out_dir) or {},
                                                       step, 1e-5))
            lrplan.append_event(out_dir, step, 1e-5, "live")
            edited = True
        time.sleep(0.2)
    wait(job)
    assert job.status == "done", "live lr run: %s" % job.message
    assert edited, "the run finished before it could be edited"

    assert abs(job.detail["lr"] - 1e-5) < 1e-12, job.detail.get("lr")
    logged = [l for l in job.detail.get("log") or [] if l.startswith("lr ")]
    assert any("1e-05" in l for l in logged), logged
    events = lrplan.read_events(out_dir)
    assert any(e["why"] == "live" for e in events), events
    assert any(e["step"] >= 30 and abs(e["lr"] - 1e-5) < 1e-12 for e in events), events

    view = load_run_view(out_dir)
    assert len(view["lr_marks"]) >= 2, view["lr_marks"]
    # the loop reads the plan and must never write it back
    disk = lrplan.read(out_dir)
    assert disk["segments"][-1]["lr"] == 1e-5, disk
    # `lr` in run.json stays a bare float: config_from_run does float() on it
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert isinstance(meta["lr"], float), meta["lr"]
    print("  live rate change took effect at step %s; %d marks on the curve"
          % (events[-1]["step"], len(view["lr_marks"])))


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
        "model_name": run_name, "mults": [1, 2, 2, 2],
        # A two-segment plan: the only end-to-end proof that the vendored step
        # hook is installed and fires inside the real subprocess.
        "lr_plan": {"segments": [{"from": 0, "kind": "const", "lr": 1e-4},
                                 {"from": 50, "kind": "const", "lr": 1e-5}]},
        **over,
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

    # the vendored step hook ran: the plan's second segment took effect on time
    from app.core.engine import lr_plan as lrplan
    logged = [l for l in (job.detail.get("log") or []) if l.startswith("lr ")]
    assert any("from step 50" in l and "1e-05" in l for l in logged), logged
    assert [e["step"] for e in lrplan.read_events(out_dir)] == [0, 50], \
        lrplan.read_events(out_dir)
    print("  learning-rate plan applied inside the subprocess:", logged)

    # and resuming the run keeps the same layout without being told it again
    from app.core.engine.trainer import config_from_run
    rcfg = config_from_run(out_dir, 200)
    assert rcfg.mtype == cfg.mtype and rcfg.attn == cfg.attn, (rcfg.mtype, rcfg.attn)
    # the plan comes back from its own file, without the caller passing it again
    assert rcfg.lr_plan == cfg.lr_plan, (rcfg.lr_plan, cfg.lr_plan)
    # A continue repeats the run's own precision rather than today's default.
    assert rcfg.amp == cfg.amp, (rcfg.amp, cfg.amp)
    rjob = wait(backend.start_training(rcfg), timeout=900)
    if rjob.status != "done":
        print("  log tail:", (rjob.detail.get("log") or [])[-8:])
    assert rjob.status == "done", f"xurdif resume: {rjob.message}"
    assert len(list_checkpoints(out_dir, 50)) >= 3, "resume added no snapshots"
    # the resume kept the first launch's log, so the curve still starts near 0
    assert (out_dir / "train.log.1").exists(), "the resume truncated train.log"
    first = min(p["step"] for p in load_run_view(out_dir)["losses"])
    assert first < 100, "the pre-resume half of the loss curve is missing"
    print("  resumed to step 200 on the same layout, curve intact from step", first)


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


def check_record_dataset(diffusers_only: bool):
    """A record dataset: non-square images in a folder outside the workspace,
    framed and augmented into a fixed set, through the real /api/train route."""
    from app.backend.app import create_app
    from app.backend.data import manifest

    external = Path(tempfile.mkdtemp(prefix="kiln_train_ext_"))
    try:
        torch.manual_seed(1)
        for i in range(10):
            small = torch.rand(3, 3, 5)
            img = torch.nn.functional.interpolate(small[None], size=(60, 100), mode="bicubic",
                                                  align_corners=False)[0]
            Image.fromarray(img.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()).save(
                external / f"wide{i:02d}.jpg")
        ds_dir = WORKSPACE / "datasets" / "linked"
        manifest.new(ds_dir, {"width": 64, "height": 64, "resize_mode": "pad",
                              "augmentations": ["vflip", "rotate", "brightness"]})
        manifest.add_path(ds_dir, str(external))
        before = sorted(p.stat().st_mtime for p in external.iterdir())

        c = create_app().test_client()
        r = c.post("/api/train", json={
            "dataset": "linked", "run_name": "drec", "backend": "diffusers", "mode": "scratch",
            "preset": "small-64", "image_size": 64, "batch_size": 4, "train_steps": 20,
            "save_every": 10, "model_name": "drec"})
        assert r.status_code == 200, r.get_json()
        from utils.process_control import registry
        job = wait(registry.get(r.get_json()["data"]["job"]["id"]))
        out_dir = WORKSPACE / "runs" / "drec"
        check_run_shape(out_dir, job, "diffusers record", expect_dirs=True)
        snap = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
        assert len(snap["files"]) == 10 and snap["recipe"]["augmentations"] == ["vflip", "rotate", "brightness"]
        assert snap["variants"] == 24 and snap["total"] == 240, snap  # 2 flips x 4 turns x 3
        log = " | ".join(job.detail.get("log") or [])
        assert "10 images x 24 versions = 240 per pass" in log, log[-400:]

        if not diffusers_only and torch.cuda.is_available():
            backend = backends.get("xurdif")
            xout = WORKSPACE / "runs" / "xrec"
            cfg = backend.training_config({
                "image_size": 64, "batch_size": 2, "train_steps": 100, "save_every": 50,
                "accum": 1, "model_name": "xrec", "mults": [1, 2, 2, 2]}, ds_dir, xout)
            cfg.manifest = str(manifest.snapshot(ds_dir, xout))
            job = wait(backend.start_training(cfg), timeout=900)
            if job.status != "done":
                print("  log tail:", (job.detail.get("log") or [])[-8:])
            check_run_shape(xout, job, "xurdif record", expect_dirs=False)
            meta = json.loads((xout / "run.json").read_text(encoding="utf-8"))
            assert meta["dataset_snapshot"] == cfg.manifest

            # the dataset record goes away; the run still resumes from its snapshot
            shutil.rmtree(ds_dir)
            from app.core.engine.trainer import config_from_run
            rcfg = config_from_run(xout, 200)
            assert rcfg.manifest == cfg.manifest
            rjob = wait(backend.start_training(rcfg), timeout=900)
            assert rjob.status == "done", f"xurdif record resume: {rjob.message}"
            print("  xurdif record run resumed from its snapshot after the dataset was deleted")
        elif not diffusers_only:
            print("  xurdif record leg skipped: no GPU")

        after = sorted(p.stat().st_mtime for p in external.iterdir())
        assert before == after and len(after) == 10, "training changed the linked images"
    finally:
        shutil.rmtree(external, ignore_errors=True)


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
        print("learning-rate schedules:")
        check_lr_plan()
        check_lr_log_merge()
        check_vram_estimate()
        check_vram_estimate_amp()
        check_amp_default()
        check_child_stream_decoding()
        print("training runs:")
        if not diffusers_only:
            check_xurdif()
            # the class every earlier model was trained with still trains
            check_xurdif("xur-old", mtype="tinyunet_with_attention3")
        base = check_diffusers_scratch()
        check_live_lr_edit()
        check_tinyunet("mse")
        check_tinyunet("xurdif")
        check_diffusers_finetune(base)
        check_diffusers_lora(base)
        print("record dataset (linked folder, every combination of its augmentations):")
        check_record_dataset(diffusers_only)
        print("capability gates:")
        check_refusals()
    finally:
        shutil.rmtree(WORKSPACE, ignore_errors=True)
    print("OK")


if __name__ == "__main__":
    main()
