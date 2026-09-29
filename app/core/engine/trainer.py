"""Training orchestration.

Runs the vendored ``xurdiftrainer.py`` as a controlled subprocess, parses its
stdout for progress (per-step loss, snapshot saves), and reports through a Job in
the process registry so the UI can stream status and stop the run. Isolating the
CUDA training loop in a subprocess keeps the API server responsive and matches the
Fragmenta approach.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.core.engine import lr_plan as lrplan
from app.core.engine._vendor import VENDOR_XURDIF
from utils.logger import get_logger
from utils.process_control import Job, registry

log = get_logger("trainer")

_STEP_RE = re.compile(r"^(\d+):\s+([\d.eE+-]+)\s*$")
_AVG_RE = re.compile(r"average loss:\s+([\d.eE+-]+)")
# What the learning-rate hook prints when the rate changes. Deliberately shaped
# so it matches neither parser above (_STEP_RE is anchored on "^<digits>:").
_LR_RE = re.compile(r"^lr\s+([\d.eE+-]+)\s+from step\s+(\d+)\s*$")

# How many previous launches' logs to keep beside train.log. A continue used to
# truncate the log, which threw away the loss history you need in order to judge
# whether a rate drop helped.
LOG_KEEP = 9

# how many recent stdout lines to keep in the (JSON-polled) live log tail
LOG_TAIL = 500

_REPO_ROOT = Path(__file__).resolve().parents[3]
_cuda_cache: dict[str, bool | None] = {}


def trainer_python() -> str:
    """Interpreter used to launch the trainer subprocess.

    Training MUST run on the project's ``.venv`` (CUDA-enabled PyTorch), even when
    the backend itself was started with a different/system Python that only has a
    CPU torch build — otherwise training silently falls back to the CPU and is
    ~100x slower. Prefer the repo venv; fall back to the current interpreter.
    """
    if os.name == "nt":
        cand = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        cand = _REPO_ROOT / ".venv" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def interpreter_cuda_report(py: str) -> dict:
    """What ``py``'s torch is and whether it can use the GPU (cached).

    ``available`` is None when the probe itself failed (no torch, or it would
    not import). The version and build come back too, so a refusal can say which
    wheel is installed rather than calling every failure "CPU-only torch".
    """
    if py in _cuda_cache:
        return _cuda_cache[py]
    report: dict = {"available": None, "version": None, "cuda": None}
    code = ("import json, torch;"
            "print(json.dumps([torch.__version__, torch.version.cuda, "
            "torch.cuda.is_available()]))")
    try:
        out = subprocess.check_output(
            [py, "-c", code], text=True, stderr=subprocess.DEVNULL, timeout=300,
        ).strip()
        version, cuda, ok = json.loads(out.splitlines()[-1])
        report = {"available": bool(ok), "version": version, "cuda": cuda}
    except Exception:  # noqa: BLE001
        pass
    _cuda_cache[py] = report
    return report


def interpreter_has_cuda(py: str) -> bool | None:
    """Whether ``py`` can see a CUDA GPU (cached; None if the probe failed)."""
    return interpreter_cuda_report(py)["available"]


@dataclass
class TrainConfig:
    dataset: str                 # folder of training images (a record dataset's own folder)
    out_dir: str                 # where checkpoints + samples are written
    name: str = "model"
    image_size: int = 512
    batch_size: int = 8
    diffusion_steps: int = 1000  # --steps
    train_steps: int = 280000    # --trainsteps (iterations)
    accum: int = 10
    lr: float = 4e-4
    # Learning-rate plan (app/core/engine/lr_plan.py), written to
    # <out_dir>/lr_plan.json before launch and re-read by the training loop when
    # it changes. ``lr`` stays a bare float and remains the base rate -- the
    # plan's value at step 0 -- so everything that already reads it is untouched.
    lr_plan: dict | None = None
    loss_type: str = "l1"
    l1w: float = 1.0
    ssimw: float = 0.0
    pred: str = "x0"
    mtype: str = "tinyunet_conf_attention"
    mults: list = field(default_factory=lambda: [1, 2, 2, 2])
    # Attention layout, canonical spec ("-1:linear,mid:full"). Only the conf
    # architecture reads it; None for the others. Passed as one token because
    # the value starts with '-'.
    attn: str | None = "-1:linear,mid:full"
    fit: str = "resize"
    # Snapshot previews: one image on a fixed seed, so a run's thumbnails differ
    # only by how far training got. -1 leaves them unseeded.
    nsamples: int = 1
    sample_seed: int = 42
    save_every: int = 1000
    amp: bool = False
    # The edge-weighted L1 the vendored engine trains with. On by default, and
    # read back as True for runs written before it was switchable -- every one
    # of those was edge-trained, so True is what they actually did.
    edge_loss: bool = True
    resume: str | None = None
    nostrict: bool = False
    continued_from: str | None = None
    # A record-based dataset's snapshot (<run>/dataset.json: file list + recipe).
    # When set the trainer reads images from it and augments as it loads them;
    # None trains from the ``dataset`` folder exactly as before.
    manifest: str | None = None

    def to_args(self) -> list[str]:
        args = [
            "--images", str(self.dataset),
            "--dir", str(self.out_dir),
            "--name", self.name,
            "--imageSize", str(self.image_size),
            "--batchSize", str(self.batch_size),
            "--steps", str(self.diffusion_steps),
            "--trainsteps", str(self.train_steps),
            "--accum", str(self.accum),
            "--lr", str(self.lr),
            "--losstype", self.loss_type,
            "--l1w", str(self.l1w),
            "--ssimw", str(self.ssimw),
            "--pred", self.pred,
            "--model", self.mtype,
            "--nsamples", str(self.nsamples),
            "--sampleSeed", str(self.sample_seed),
            "--saveEvery", str(self.save_every),
            "--fit", self.fit,
            "--mults", *[str(m) for m in self.mults],
        ]
        if self.attn:
            args.append(f"--attn={self.attn}")
        if self.manifest:
            args += ["--manifest", str(self.manifest)]
        # The path, not the schedule: because the plan is a file, choosing a curve
        # up front and dropping the rate mid-run are the same code path. Passed
        # unconditionally -- a constant run gets a one-segment plan -- so there is
        # one path in the loop and "drop it now" works on any run.
        args += ["--lrPlan", str(Path(self.out_dir) / lrplan.PLAN_NAME)]
        if self.amp:
            args.append("--amp")
        if not self.edge_loss:
            # Opt-out, matching the vendored flag: absent means the edge-weighted
            # L1 that every model before this option was trained with.
            args.append("--noEdges")
        if self.resume:
            args += ["--load", self.resume]
            if self.nostrict:
                args.append("--nostrict")
        return args


def config_from_run(run_dir: str | Path, train_steps: int, checkpoint: str | None = None) -> TrainConfig:
    """Build a TrainConfig to resume training in an existing run folder."""
    from utils.exceptions import ValidationError

    run_dir = Path(run_dir)
    meta = _read_run_meta(run_dir)
    if not meta:
        raise ValidationError("run metadata missing")
    save_every = meta.get("save_every")
    ckpts = list_checkpoints(run_dir, save_every)
    if not ckpts:
        raise ValidationError("no checkpoints in this run")
    if checkpoint:
        ckpt = next((c for c in ckpts if c["filename"] == checkpoint or c["path"] == checkpoint), None)
        if not ckpt:
            raise ValidationError("checkpoint not found in this run")
    else:
        ckpt = ckpts[-1]
    ckpt_step = ckpt.get("step") or 0
    if train_steps <= ckpt_step:
        raise ValidationError(f"train_steps must be greater than current step ({ckpt_step})")
    dataset = meta.get("dataset")
    manifest = meta.get("dataset_snapshot") or None
    if manifest:
        # A record dataset resumes from the file list it trained on, so editing
        # or deleting the dataset since does not change what the run continues on.
        if not Path(manifest).exists():
            raise ValidationError("this run's dataset snapshot is missing")
        from app.core.engine.train_data import read_snapshot

        if not any(Path(f).exists() for f in read_snapshot(manifest)["files"]):
            raise ValidationError("none of the images this run trained on can be found any more")
    elif not dataset or not Path(dataset).exists():
        raise ValidationError("original dataset for this run is missing")
    mults = parse_mults(meta.get("mults"))
    return TrainConfig(
        dataset=str(dataset),
        out_dir=str(run_dir),
        name=meta.get("name") or run_dir.name,
        image_size=int(meta.get("image_size", 512)),
        batch_size=int(meta.get("batch_size", 8)),
        diffusion_steps=int(meta.get("diffusion_steps", 1000)),
        train_steps=int(train_steps),
        accum=int(meta.get("accum", 10)),
        lr=float(meta.get("lr", 4e-4)),
        # From the plan file, not run.json: a live edit writes the file, and
        # run.json's read-modify-write races the trainer thread.
        lr_plan=lrplan.read(run_dir),
        loss_type=meta.get("loss_type", "l1"),
        l1w=float(meta.get("l1w", 1.0)),
        ssimw=float(meta.get("ssimw", 0.0)),
        pred=meta.get("pred", "x0"),
        # A run.json always names its mtype; the fallback is for one written
        # before the field existed, which was necessarily the old class.
        mtype=meta.get("mtype", "tinyunet_with_attention3"),
        mults=mults,
        attn=meta.get("attn"),
        fit=meta.get("fit", "resize"),
        nsamples=int(meta.get("nsamples", 1)),
        sample_seed=int(meta.get("sample_seed", 42)),
        save_every=int(save_every or 1000),
        amp=bool(meta.get("amp", False)),
        edge_loss=bool(meta.get("edge_loss", True)),
        resume=ckpt["path"],
        continued_from=ckpt["filename"],
        manifest=manifest,
    )


def _latest_sample(out_dir: Path) -> str | None:
    samples = sorted(out_dir.glob("sample-*.png"), key=lambda p: p.stat().st_mtime)
    return str(samples[-1]) if samples else None


def list_checkpoints(out_dir: str | Path, save_every: int | None = None) -> list[dict]:
    """Enumerate ``model-*.pt`` snapshots written by a run, newest last.

    ``step`` is derived from ``milestone * save_every`` (xurdif saves at
    ``step // save_every``), so we avoid loading the (large) checkpoint files.
    """
    out_dir = Path(out_dir)
    if save_every is None:
        save_every = _read_run_meta(out_dir).get("save_every")
    items: list[dict] = []
    # A xurdif snapshot is one .pt file; a Diffusers snapshot is a model-N/
    # directory written by save_pretrained. Both are "the checkpoint at
    # milestone N" as far as every screen that lists them is concerned.
    for entry in out_dir.glob("model-*"):
        is_dir = entry.is_dir()
        if not is_dir and entry.suffix != ".pt":
            continue
        try:
            milestone = int(entry.stem.split("-", 1)[1])
        except (ValueError, IndexError):
            milestone = None
        st = entry.stat()
        if is_dir:
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            mtime = max((f.stat().st_mtime for f in entry.rglob("*") if f.is_file()),
                        default=st.st_mtime)
        else:
            size, mtime = st.st_size, st.st_mtime
        sample = out_dir / f"sample-{milestone}.png" if milestone is not None else None
        items.append({
            "filename": entry.name,
            "path": str(entry),
            "milestone": milestone,
            "step": milestone * save_every if (milestone is not None and save_every) else None,
            "size_mb": round(size / (1024 * 1024), 2),
            "mtime": mtime,
            "sample": str(sample) if (sample and sample.exists()) else None,
        })
    items.sort(key=lambda c: c["mtime"])
    return items


# The measured fit below was taken at mults 1,2,2,2, whose activation profile is
# this sum; dividing by it normalises any other width set against that baseline.
_BASE_MULTS = (1, 2, 2, 2)
_FIXED_MIB = 825.0          # CUDA context + framework, with the weights term split out
_PER_IMG_512_MIB = 715.0    # activations per 512px image at _BASE_MULTS
_BYTES_PER_PARAM = 20       # fp32 weights + grads + two Adam moments + the EMA copy
_param_cache: dict[tuple, int] = {}


def parse_mults(raw) -> list[int]:
    """Channel multipliers as a list of ints, from a list or a "1,2,2,4" string.

    Shape only -- ``backends.xurdif._clean_mults`` is the validating parser, which
    also checks the image size divides by 2**len(mults). This one exists because
    mults arrive from query strings and from run.json in both spellings.
    """
    if not raw:
        return list(_BASE_MULTS)
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(",") if x.strip()]
    try:
        out = [int(m) for m in raw]
    except (TypeError, ValueError):
        raise ValueError(f"channel multipliers must be whole numbers, got {raw!r}")
    if not out or any(m < 1 for m in out):
        raise ValueError(f"channel multipliers must be positive, got {raw!r}")
    return out


def _act_scale(mults) -> float:
    """How activation memory scales with width, relative to mults 1,2,2,2.

    Level *i* holds ``dim * m_i`` channels at ``size / 2**i`` on a side, so its
    cost goes as ``m_i / 4**i``. Deep levels are spatially tiny, which is why
    widening the tail barely moves activations at all -- the real cost of a wider
    net is the optimizer state below and the step time.
    """
    def total(ms):
        return sum(float(m) / (4.0 ** i) for i, m in enumerate(ms)) or 1.0
    return total(parse_mults(mults)) / total(_BASE_MULTS)


def param_count(mults, mtype: str | None = None, attn: str | None = None) -> int:
    """Parameters in the net these settings build. Built on the CPU and memoised:
    construction is pure ``torch.nn`` and takes no measurable time."""
    from app.core.engine.arch import CONF_MTYPE

    mults = tuple(parse_mults(mults))
    mtype = mtype or CONF_MTYPE
    key = (mults, mtype, attn)
    if key in _param_cache:
        return _param_cache[key]
    try:
        from app.core.backends.xurdif import attn as attn_spec
        from app.core.engine.arch import build_unet

        cfg = attn_spec.parse(attn) if mtype == CONF_MTYPE else None
        net = build_unet(mtype, mults, attn_config=cfg)
        n = sum(p.numel() for p in net.parameters())
    except Exception as e:  # noqa: BLE001
        # Only reachable if torch or the vendored snapshot is unusable, in which
        # case training cannot run either; fall back to the baseline net's size.
        log.warning("could not count parameters for mults %s: %s", list(mults), e)
        n = 3_759_043          # the baseline net at mults 1,2,2,2
    _param_cache[key] = n
    return n


def _weights_mib(mults, mtype=None, attn=None) -> float:
    return param_count(mults, mtype, attn) * _BYTES_PER_PARAM / (1024.0 * 1024.0)


def estimate_peak_mib(image_size: int, batch_size: int, mults=None,
                      mtype: str | None = None, attn: str | None = None) -> int:
    """Rough peak-VRAM estimate for the tinyunet trainer.

    Fitted from measured peaks on an RTX 3060 (dim=64, mults 1,2,2,2, AMP):
    activation memory scales ~ (size/512)^2 * 715 MiB per image. The rest is the
    CUDA context plus weights, gradients, Adam's two moments and the EMA copy --
    which is where a wider network actually costs, since its extra activations
    sit at the spatially smallest levels.
    """
    per_img = _PER_IMG_512_MIB * (image_size / 512.0) ** 2 * _act_scale(mults)
    return int(per_img * max(batch_size, 1) + _FIXED_MIB + _weights_mib(mults, mtype, attn))


def recommended_batch(image_size: int, total_mib: int, mults=None,
                      mtype: str | None = None, attn: str | None = None) -> int:
    """Largest batch that should fit comfortably (avoids sysmem-fallback cliff)."""
    per_img = _PER_IMG_512_MIB * (image_size / 512.0) ** 2 * _act_scale(mults)
    budget = total_mib * 0.82 - _FIXED_MIB - _weights_mib(mults, mtype, attn)
    return max(1, int(budget / per_img)) if per_img > 0 else 1


def _rotate_run_log(out_dir: Path) -> Path:
    """Shift train.log to train.log.1 (and so on) so a relaunch does not destroy
    the previous one.

    A continue used to open train.log with "w". Since load_run_view rebuilds the
    loss curve by parsing the log, that threw away every point before the resume
    -- which is exactly the half of the chart you need in order to see whether a
    learning-rate drop helped.
    """
    live = out_dir / "train.log"
    if not live.exists():
        return live
    try:
        oldest = out_dir / f"train.log.{LOG_KEEP}"
        if oldest.exists():
            oldest.unlink()
        for n in range(LOG_KEEP - 1, 0, -1):
            src = out_dir / f"train.log.{n}"
            if src.exists():
                src.replace(out_dir / f"train.log.{n + 1}")
        live.replace(out_dir / "train.log.1")
    except OSError as e:
        log.warning("could not rotate train.log: %s", e)
    return live


def _log_segments(out_dir: Path) -> list[list[str]]:
    """Every launch's log lines, oldest launch first."""
    paths = [out_dir / f"train.log.{n}" for n in range(LOG_KEEP, 0, -1)]
    paths.append(out_dir / "train.log")
    segs: list[list[str]] = []
    for f in paths:
        if not f.exists():
            continue
        try:
            segs.append(f.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
    return segs


def _read_run_meta(out_dir: Path) -> dict:
    f = out_dir / "run.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def run_meta(out_dir: str | Path) -> dict:
    """A run's run.json, or {} if it has none."""
    return _read_run_meta(Path(out_dir))


def _loss_stride(save_every) -> int:
    """How often to keep a loss point. Shared by the live and on-disk paths so
    a run's chart looks the same while it trains and after it finishes."""
    return max(min(int(save_every or 0) // 2, 50), 1)


def _write_run_meta(out_dir: Path, cfg: "TrainConfig", *, continued_from: str | None = None):
    existing = _read_run_meta(out_dir)
    meta = {
        **existing,
        "name": cfg.name,
        "dataset": cfg.dataset,
        "dataset_snapshot": cfg.manifest or "",
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "train_steps": cfg.train_steps,
        "save_every": cfg.save_every,
        "mtype": cfg.mtype,
        "mults": cfg.mults,
        "attn": cfg.attn,
        "pred": cfg.pred,
        "lr": cfg.lr,
        # Display only, and stale-tolerant: the plan itself lives in its own file
        # so a live rewrite cannot race this read-modify-write.
        **({"lr_schedule": (cfg.lr_plan or {}).get("preset") or "custom",
            "lr_schedule_summary": lrplan.summarize(cfg.lr_plan)} if cfg.lr_plan else {}),
        "loss_type": cfg.loss_type,
        "l1w": cfg.l1w,
        "ssimw": cfg.ssimw,
        "accum": cfg.accum,
        "diffusion_steps": cfg.diffusion_steps,
        "fit": cfg.fit,
        "nsamples": cfg.nsamples,
        "sample_seed": cfg.sample_seed,
        "amp": cfg.amp,
        "edge_loss": cfg.edge_loss,
        "status": "training",
        "resume": cfg.resume or "",
    }
    if continued_from:
        meta["continued_from"] = continued_from
        continues = list(meta.get("continues") or [])
        continues.append({"checkpoint": continued_from, "at": time.time()})
        meta["continues"] = continues
    try:
        (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("could not write run.json: %s", e)


def patch_run_meta(out_dir: str | Path, **fields):
    out_dir = Path(out_dir)
    meta = _read_run_meta(out_dir)
    meta.update(fields)
    try:
        (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("could not update run.json: %s", e)


def load_run_view(out_dir: str | Path) -> dict:
    """Disk snapshot of a run: settings, checkpoints, log tail, parsed loss, last sample."""
    out_dir = Path(out_dir)
    meta = _read_run_meta(out_dir)
    save_every = meta.get("save_every")
    ckpts = list_checkpoints(out_dir, save_every) if out_dir.exists() else []
    log_path = out_dir / "train.log"
    segments = _log_segments(out_dir) if out_dir.exists() else []
    # The log panel shows a tail of the *current* launch. The loss curve spans
    # every launch, so it survives a continue.
    log_lines = segments[-1][-LOG_TAIL:] if segments else []

    def _points(lines: list[str]) -> list[dict]:
        pts = []
        for line in lines:
            m = _STEP_RE.match(line)
            if not m:
                continue
            step = int(m.group(1))
            # Sampled, not every line -- but capped at 50 so curve density stops
            # riding on snapshot frequency. At save_every 1000 the old half-of-it
            # rule plotted a point every 500 steps, which is a chart of four dots.
            if save_every and step % _loss_stride(save_every) != 0:
                continue
            pts.append({"step": step, "loss": float(m.group(2))})
        return pts

    # Newest launch first, keeping only steps no later launch already describes.
    # Continuing from an *earlier* checkpoint therefore supersedes the overlapping
    # tail of the run it replaced, instead of drawing a zigzag.
    chunks: list[list[dict]] = []
    cutoff = None
    for lines in reversed(segments):
        pts = _points(lines)
        if cutoff is not None:
            pts = [q for q in pts if q["step"] < cutoff]
        if pts:
            cutoff = pts[0]["step"]
            chunks.append(pts)
    losses = [q for c in reversed(chunks) for q in c]
    if len(losses) > 240:
        step_n = max(len(losses) // 240, 1)
        losses = losses[::step_n]
    plan = lrplan.read(out_dir)
    return {
        "name": out_dir.name,
        "path": str(out_dir),
        "meta": meta,
        "status": meta.get("status") or ("stopped" if ckpts else "not_started"),
        "checkpoints": ckpts,
        "log": log_lines,
        "losses": losses,
        "lr_plan": plan,
        "lr_marks": lrplan.marks_for_chart(lrplan.read_events(out_dir), plan),
        "sample": _latest_sample(out_dir),
        "log_path": str(log_path) if log_path.exists() else None,
    }


def _run(job: Job, cfg: TrainConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_run_meta(out_dir, cfg, continued_from=cfg.continued_from)
    plan = cfg.lr_plan or lrplan.compile_plan(
        None, lr=cfg.lr, train_steps=cfg.train_steps, save_every=cfg.save_every)
    lrplan.write(out_dir, plan)
    job.detail["lr_plan"] = plan
    job.detail["lr"] = lrplan.lr_at(plan, 0)
    py = trainer_python()
    # -u because the child's prints otherwise sit in an 8KB pipe buffer for some
    # 300 steps, which delays the live loss curve and makes "the new rate applies
    # on the next step" impossible to see.
    cmd = [py, "-u", "xurdiftrainer.py", *cfg.to_args()]
    job.detail["cmd"] = " ".join(cmd)
    job.detail["python"] = py
    job.detail["out_dir"] = str(out_dir)
    job.detail["log_path"] = str(out_dir / "train.log")
    job.detail["checkpoints"] = list_checkpoints(out_dir, cfg.save_every)
    log.info("launching trainer with %s", py)

    # verify the chosen interpreter can actually use the GPU (xurdif hardcodes
    # .cuda(); a CPU-only torch would either crash or crawl).
    cuda = interpreter_has_cuda(py)
    job.detail["cuda"] = cuda
    if cuda is False:
        from utils import cuda as cuda_pick

        report = interpreter_cuda_report(py)
        why = cuda_pick.diagnose(report["version"], report["cuda"],
                                 cuda_pick.driver_version(), python=py)
        job.status = "error"
        job.message = f"training needs a GPU, and this Python cannot use one. {why['hint']}"
        log.error(job.message)
        patch_run_meta(out_dir, status="error", message=job.message, finished_at=time.time())
        return
    if cuda is None:
        log.warning("could not verify CUDA for %s; launching anyway", py)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(VENDOR_XURDIF),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:  # noqa: BLE001
        job.status = "error"
        job.message = f"failed to launch trainer: {e}"
        patch_run_meta(out_dir, status="error", message=job.message, finished_at=time.time())
        return

    job.detail["pid"] = proc.pid
    # allow the cancel route to kill the subprocess immediately, even while it is
    # still starting up and has produced no output yet.
    job.stopper = lambda: _stop_proc(proc)
    losses = job.detail.setdefault("losses", [])
    logbuf: list[str] = job.detail.setdefault("log", [])
    pending_ckpt_refresh = False

    try:
        logf = open(_rotate_run_log(out_dir), "w", encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        logf = None

    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        if not line:
            continue

        # --- log capture (full file + bounded live tail for the UI panel) ---
        if logf:
            logf.write(line + "\n")
            logf.flush()
        logbuf.append(line)
        if len(logbuf) > LOG_TAIL:
            del logbuf[: len(logbuf) - LOG_TAIL]

        job.detail["last_line"] = line
        m = _STEP_RE.match(line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            job.progress = min(step / max(cfg.train_steps, 1), 0.999)
            job.message = f"step {step} / {cfg.train_steps}  loss {loss:.4f}"
            job.detail["step"] = step
            job.detail["loss"] = loss
            if step % _loss_stride(cfg.save_every) == 0:
                losses.append({"step": step, "loss": loss})
            samp = _latest_sample(out_dir)
            if samp:
                job.detail["sample"] = samp
            # a checkpoint is flushed right before the *next* step line prints,
            # so refresh the list now that the snapshot file exists on disk.
            if pending_ckpt_refresh:
                job.detail["checkpoints"] = list_checkpoints(out_dir, cfg.save_every)
                pending_ckpt_refresh = False
        elif _AVG_RE.search(line):
            job.detail["avg_loss_line"] = line
            pending_ckpt_refresh = True
        elif (lm := _LR_RE.match(line)):
            at, rate = int(lm.group(2)), float(lm.group(1))
            job.detail["lr"] = rate
            lrplan.append_event(out_dir, at, rate)
            job.detail["lr_plan"] = lrplan.read(out_dir)
            job.detail["lr_marks"] = lrplan.marks_for_chart(
                lrplan.read_events(out_dir), job.detail["lr_plan"])

        if job.cancelled():
            _stop_proc(proc)
            break

    if logf:
        logf.close()
    proc.wait()
    # final sweep so the last snapshot is always listed
    job.detail["checkpoints"] = list_checkpoints(out_dir, cfg.save_every)
    if job.cancelled():
        job.status = "cancelled"
        job.message = "training stopped"
    elif job.status == "running":
        if proc.returncode == 0:
            job.status = "done"
            job.progress = 1.0
            job.message = "training completed"
        else:
            job.status = "error"
            job.message = f"trainer exited with code {proc.returncode}: {job.detail.get('last_line','')}"
    patch_run_meta(
        out_dir,
        status=job.status,
        message=job.message,
        step=job.detail.get("step"),
        finished_at=time.time(),
    )


def _stop_proc(proc: subprocess.Popen):
    """Terminate the trainer subprocess without blocking the caller.

    ``terminate()`` returns immediately; a short daemon watchdog hard-kills the
    process if it hasn't exited, so a stuck/starting run is always stoppable.
    """
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:  # noqa: BLE001
        pass

    def _kill_if_stuck():
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_kill_if_stuck, daemon=True).start()


def start_training(cfg: TrainConfig) -> Job:
    job = registry.create("train")
    job.message = "starting..."
    # surface the exact command + run dir on the very first poll (before the
    # worker thread has spun up), so the UI can show it during startup.
    py = trainer_python()
    job.detail["cmd"] = " ".join([py, "-u", "xurdiftrainer.py", *cfg.to_args()])
    job.detail["python"] = py
    job.detail["out_dir"] = str(Path(cfg.out_dir))
    t = threading.Thread(target=_run, args=(job, cfg), daemon=True)
    job.thread = t
    t.start()
    return job
