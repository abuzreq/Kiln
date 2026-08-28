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

from app.core.engine._vendor import VENDOR_XURDIF
from utils.logger import get_logger
from utils.process_control import Job, registry

log = get_logger("trainer")

_STEP_RE = re.compile(r"^(\d+):\s+([\d.eE+-]+)\s*$")
_AVG_RE = re.compile(r"average loss:\s+([\d.eE+-]+)")

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


def interpreter_has_cuda(py: str) -> bool | None:
    """Return whether ``py`` can see a CUDA GPU (cached; None if the probe fails)."""
    if py in _cuda_cache:
        return _cuda_cache[py]
    val: bool | None
    try:
        out = subprocess.check_output(
            [py, "-c", "import torch;print(torch.cuda.is_available())"],
            text=True, stderr=subprocess.DEVNULL, timeout=120,
        ).strip()
        val = out.splitlines()[-1].strip() == "True" if out else None
    except Exception:  # noqa: BLE001
        val = None
    _cuda_cache[py] = val
    return val


@dataclass
class TrainConfig:
    dataset: str                 # folder of training images
    out_dir: str                 # where checkpoints + samples are written
    name: str = "model"
    image_size: int = 512
    batch_size: int = 8
    diffusion_steps: int = 1000  # --steps
    train_steps: int = 280000    # --trainsteps (iterations)
    accum: int = 10
    lr: float = 4e-4
    loss_type: str = "l1"
    l1w: float = 1.0
    ssimw: float = 0.0
    pred: str = "x0"
    mtype: str = "tinyunet_with_attention3"
    mults: list = field(default_factory=lambda: [1, 2, 2, 2])
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
    if not dataset or not Path(dataset).exists():
        raise ValidationError("original dataset for this run is missing")
    mults = meta.get("mults") or [1, 2, 2, 2]
    if isinstance(mults, str):
        mults = [int(x.strip()) for x in mults.split(",") if x.strip()]
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
        loss_type=meta.get("loss_type", "l1"),
        l1w=float(meta.get("l1w", 1.0)),
        ssimw=float(meta.get("ssimw", 0.0)),
        pred=meta.get("pred", "x0"),
        mtype=meta.get("mtype", "tinyunet_with_attention3"),
        mults=mults,
        fit=meta.get("fit", "resize"),
        nsamples=int(meta.get("nsamples", 1)),
        sample_seed=int(meta.get("sample_seed", 42)),
        save_every=int(save_every or 1000),
        amp=bool(meta.get("amp", False)),
        edge_loss=bool(meta.get("edge_loss", True)),
        resume=ckpt["path"],
        continued_from=ckpt["filename"],
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


def estimate_peak_mib(image_size: int, batch_size: int) -> int:
    """Rough peak-VRAM estimate for the default tinyunet trainer.

    Fitted from measured peaks on an RTX 3060 (tinyunet_with_attn3, dim=64,
    mults 1,2,2,2, AMP): activation memory scales ~ (size/512)^2 * 715 MiB per
    image, plus fixed overhead for weights, Adam state and the CUDA context.
    """
    per_img = 715.0 * (image_size / 512.0) ** 2
    return int(per_img * max(batch_size, 1) + 900.0)


def recommended_batch(image_size: int, total_mib: int) -> int:
    """Largest batch that should fit comfortably (avoids sysmem-fallback cliff)."""
    per_img = 715.0 * (image_size / 512.0) ** 2
    budget = total_mib * 0.82 - 900.0
    return max(1, int(budget / per_img)) if per_img > 0 else 1


def _read_run_meta(out_dir: Path) -> dict:
    f = out_dir / "run.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


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
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "train_steps": cfg.train_steps,
        "save_every": cfg.save_every,
        "mtype": cfg.mtype,
        "mults": cfg.mults,
        "pred": cfg.pred,
        "lr": cfg.lr,
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
    log_lines: list[str] = []
    all_lines: list[str] = []
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            all_lines = text.splitlines()
            log_lines = all_lines[-LOG_TAIL:]
        except Exception:  # noqa: BLE001
            all_lines = []
            log_lines = []
    # the log panel shows a tail, but the loss curve is parsed from the whole
    # file so it spans every step of the run, not just the last few hundred lines.
    losses = []
    for line in all_lines:
        m = _STEP_RE.match(line)
        if not m:
            continue
        step = int(m.group(1))
        # Sampled, not every line -- but capped at 50 so curve density stops
        # riding on snapshot frequency. At save_every 1000 the old half-of-it
        # rule plotted a point every 500 steps, which is a chart of four dots.
        if save_every and step % _loss_stride(save_every) != 0:
            continue
        losses.append({"step": step, "loss": float(m.group(2))})
    if len(losses) > 240:
        step_n = max(len(losses) // 240, 1)
        losses = losses[::step_n]
    return {
        "name": out_dir.name,
        "path": str(out_dir),
        "meta": meta,
        "status": meta.get("status") or ("stopped" if ckpts else "not_started"),
        "checkpoints": ckpts,
        "log": log_lines,
        "losses": losses,
        "sample": _latest_sample(out_dir),
        "log_path": str(log_path) if log_path.exists() else None,
    }


def _run(job: Job, cfg: TrainConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_run_meta(out_dir, cfg, continued_from=cfg.continued_from)
    py = trainer_python()
    cmd = [py, "xurdiftrainer.py", *cfg.to_args()]
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
        job.status = "error"
        job.message = (
            "selected Python has no CUDA GPU (CPU-only torch). Install a CUDA "
            f"build into the project .venv and retry. Interpreter: {py}"
        )
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
        logf = open(out_dir / "train.log", "w", encoding="utf-8", errors="replace")
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
    job.detail["cmd"] = " ".join([py, "xurdiftrainer.py", *cfg.to_args()])
    job.detail["python"] = py
    job.detail["out_dir"] = str(Path(cfg.out_dir))
    t = threading.Thread(target=_run, args=(job, cfg), daemon=True)
    job.thread = t
    t.start()
    return job
