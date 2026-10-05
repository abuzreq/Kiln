"""Training, fine-tuning and LoRA for Diffusers models.

Unlike xurdif -- which Kiln drives as a CUDA-only subprocess and reads back
through stdout -- this loop runs in-process on a worker thread. It still has to
look identical from the outside: the same ``Job`` fields the UI polls, the same
``run.json``, the same ``sample-N.png`` snapshots, so the Train and Models
screens work without knowing which engine produced a run.

Three modes:

- ``scratch``   -- a new UNet2DModel from a size preset
- ``finetune``  -- continue from an existing model, all weights trainable
- ``lora``      -- continue from an existing model, training only PEFT adapters

Checkpoints are written as ``model-N/`` directories (``save_pretrained``), which
is what makes them loadable by the rest of the app -- and by anything else in
the ecosystem -- rather than a private tensor dump.
"""
import json
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from utils.exceptions import ValidationError

from . import objectives
from utils.logger import get_logger
from utils.process_control import Job, registry

log = get_logger("diffusers.training")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# LoRA targets, verified present on UNet2DModel: attention projections plus the
# resnet convolutions and their time conditioning. Attention alone barely moves
# an unconditional model -- most of its capacity is in the resnets.
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "conv1", "conv2", "time_emb_proj"]

# Architecture presets for training from scratch. Kept few and named for what
# they cost, because "pick your block_out_channels" is not a question a user of
# this app should have to answer.
SCRATCH_PRESETS = {
    # The xurdif network, re-homed (see tinyunet.py). Measured at 512px it is
    # ~4x faster to sample and ~7x faster per training step than the closest
    # UNet2DModel, on less than half the training memory -- which is why it is
    # the recommended starting point on a laptop GPU, not a legacy option.
    "tiny-256": {
        "architecture": "tinyunet", "dim": 64, "dim_mults": (1, 2, 2, 2),
        "sample_size": 256,
        "label": "Tiny 256 (xurdif architecture, fastest)",
    },
    "tiny-512": {
        "architecture": "tinyunet", "dim": 64, "dim_mults": (1, 2, 2, 2),
        "sample_size": 512,
        "label": "Tiny 512 (xurdif architecture, recommended)",
    },
    "small-64": {
        "architecture": "unet2d",
        "sample_size": 64, "block_out_channels": (64, 128, 128),
        "down_block_types": ("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        "layers_per_block": 2,
        "label": "Small 64 (fast, for trying things out)",
    },
    "standard-128": {
        "architecture": "unet2d",
        "sample_size": 128, "block_out_channels": (128, 128, 256, 256),
        "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        "layers_per_block": 2,
        "label": "Standard 128 (recommended)",
    },
    "large-256": {
        "architecture": "unet2d",
        "sample_size": 256, "block_out_channels": (128, 128, 256, 256, 512, 512),
        "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D",
                             "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                           "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        "layers_per_block": 2,
        "label": "Large 256 (matches the DDPM reference models)",
    },
}


@dataclass
class DiffusersTrainConfig:
    dataset: str
    out_dir: str
    name: str = "model"
    mode: str = "scratch"                 # scratch | finetune | lora
    base_model: str | None = None         # required for finetune / lora
    preset: str = "tiny-512"              # scratch only
    # "mse" is the stock Diffusers target; "xurdif" is the edge-weighted L1 (+
    # optional SSIM) in x0 space that the vendored trainer uses, ported in
    # objectives.py and verified against it bit for bit.
    objective: str = "mse"
    l1w: float = 1.0
    ssimw: float = 0.0
    image_size: int = 128
    batch_size: int = 4
    train_steps: int = 20000
    accum: int = 1
    lr: float = 1e-4
    # Learning-rate plan; see app/core/engine/lr_plan.py. None means the constant
    # plan, which is what `lr` alone already meant. to_meta() copies __dict__, so
    # this lands in run.json without any extra plumbing.
    lr_plan: dict | None = None
    save_every: int = 500
    diffusion_steps: int = 1000
    seed: int = 42
    sample_seed: int = 42
    nsamples: int = 1
    fit: str = "resize"
    # acceleration -- every one of these is optional and falls back cleanly
    precision: str = "no"                 # no | fp16 | bf16
    gradient_checkpointing: bool = False
    compile_model: bool = False
    # The snapshots are the averaged weights; see _Ema.
    use_ema: bool = True
    ema_decay: float = 0.9995
    # AdamW's own 0.01 for a new model. A fine-tune or adapter gets 0: decay
    # pulls weights toward zero, not toward the base model, which is the
    # opposite of what "stay close" asks for.
    weight_decay: float = 0.01
    # lora only
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: list = field(default_factory=lambda: list(LORA_TARGETS))
    # A record-based dataset's snapshot; see TrainConfig.manifest in engine/trainer.py.
    manifest: str | None = None

    def to_meta(self) -> dict:
        d = dict(self.__dict__)
        d["backend"] = "diffusers"
        return d


def presets() -> dict:
    return {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                for kk, vv in v.items()} for k, v in SCRATCH_PRESETS.items()}


def run_defaults() -> dict:
    """The Train form's starting numbers on this engine, from the config's own
    defaults so the two cannot drift apart."""
    d = DiffusersTrainConfig(dataset="", out_dir="")
    return {"lr": d.lr, "batch_size": d.batch_size, "accum": d.accum,
            "train_steps": d.train_steps, "save_every": d.save_every}


def config_from_body(body: dict, dataset: str, out_dir: str) -> DiffusersTrainConfig:
    """Validate a request into a config, refusing combinations we cannot honour."""
    from utils.validators import as_float, as_int

    mode = (body.get("mode") or "scratch").lower()
    if mode not in ("scratch", "finetune", "lora"):
        raise ValidationError(f"unknown training mode: {mode}")
    base = body.get("base_model") or body.get("resume") or None
    if mode in ("finetune", "lora") and not base:
        raise ValidationError(f"'{mode}' needs a base model to start from")

    preset = body.get("preset") or "tiny-512"
    if mode == "scratch" and preset not in SCRATCH_PRESETS:
        raise ValidationError(
            f"unknown preset '{preset}' (have: {', '.join(SCRATCH_PRESETS)})")

    precision = (body.get("precision") or "no").lower()
    if precision not in ("no", "fp16", "bf16"):
        raise ValidationError(f"unknown precision: {precision}")

    objective = (body.get("objective") or "mse").lower()
    if objective not in ("mse", "xurdif"):
        raise ValidationError(f"unknown objective: {objective}")

    default_size = SCRATCH_PRESETS.get(preset, {}).get("sample_size", 128)
    return DiffusersTrainConfig(
        dataset=str(dataset),
        out_dir=str(out_dir),
        name=body.get("model_name") or body.get("name") or "model",
        mode=mode,
        base_model=base,
        preset=preset,
        objective=objective,
        l1w=as_float(body.get("l1w", 1.0), "l1w", 0.0, 100.0),
        ssimw=as_float(body.get("ssimw", 0.0), "ssimw", 0.0, 100.0),
        image_size=as_int(body.get("image_size", default_size), "image_size", 32, 1024),
        batch_size=as_int(body.get("batch_size", 4), "batch_size", 1, 64),
        train_steps=as_int(body.get("train_steps", 20000), "train_steps", 10, 5_000_000),
        accum=as_int(body.get("accum", 1), "accum", 1, 128),
        lr=as_float(body.get("lr", 1e-4), "lr", 1e-7, 1.0),
        lr_plan=_lr_plan_from(body,
                              lr=as_float(body.get("lr", 1e-4), "lr", 1e-7, 1.0),
                              train_steps=as_int(body.get("train_steps", 20000),
                                                 "train_steps", 10, 5_000_000),
                              save_every=as_int(body.get("save_every", 500),
                                                "save_every", 10, 100000)),
        save_every=as_int(body.get("save_every", 500), "save_every", 10, 100000),
        diffusion_steps=as_int(body.get("diffusion_steps", 1000), "diffusion_steps", 10, 4000),
        seed=as_int(body.get("seed", 42), "seed", 0, 2 ** 31 - 1),
        sample_seed=as_int(body.get("sample_seed", 42), "sample_seed", -1, 2 ** 31 - 1),
        nsamples=as_int(body.get("nsamples", 1), "nsamples", 1, 4),
        fit=body.get("fit", "resize"),
        precision=precision,
        gradient_checkpointing=bool(body.get("gradient_checkpointing", False)),
        compile_model=bool(body.get("compile", False)),
        use_ema=bool(body.get("use_ema", True)),
        weight_decay=as_float(body.get("weight_decay", 0.0 if mode in ("finetune", "lora") else 0.01),
                              "weight_decay", 0.0, 1.0),
        lora_r=as_int(body.get("lora_r", 8), "lora_r", 1, 256),
        lora_alpha=as_int(body.get("lora_alpha", 16), "lora_alpha", 1, 512),
        lora_dropout=as_float(body.get("lora_dropout", 0.0), "lora_dropout", 0.0, 0.9),
        lora_targets=list(body.get("lora_targets") or LORA_TARGETS),
    )


class _Ema:
    """A running average of the trainable weights, which is what snapshots save.

    The raw weights at any one step carry that step's noise; the average is the
    smoother model, and the one an xurdif snapshot has always been sampled
    from. The default decay, 0.9995 per update, is the vendored trainer's 0.995
    every 10 updates. It ramps up from nothing, (1+n)/(10+n), so a short
    fine-tune's average is not still mostly the base model it started from.
    """

    def __init__(self, params, decay: float):
        import torch

        self.params = [p for p in params if p.requires_grad]
        self.decay = float(decay)
        self.n = 0
        with torch.no_grad():
            self.shadow = [p.detach().float().clone() for p in self.params]

    def update(self):
        import torch

        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        with torch.no_grad():
            for avg, p in zip(self.shadow, self.params):
                avg.lerp_(p.detach().float(), 1.0 - d)

    @contextmanager
    def applied(self):
        """The averaged weights in the model for the duration; the raw ones,
        which training carries on from, put back afterwards."""
        import torch

        with torch.no_grad():
            raw = [p.detach().clone() for p in self.params]
            for avg, p in zip(self.shadow, self.params):
                p.copy_(avg)
        try:
            yield
        finally:
            with torch.no_grad():
                for r, p in zip(raw, self.params):
                    p.copy_(r)


def _lr_plan_from(body: dict, *, lr: float, train_steps: int, save_every: int) -> dict:
    """The schedule this request asks for, compiled to absolute steps."""
    from app.core.engine import lr_plan as lrplan

    return lrplan.new_run_plan(body, lr=lr, train_steps=train_steps, save_every=save_every)


class ImageFolder:
    """Every image under a folder, resized or cropped into [-1, 1].

    [-1, 1] is the Diffusers convention and what Kiln's sampler already assumes
    on the way back out. (xurdif trains in [-0.5, 0.5] and leans on its display
    renormalisation to compensate -- not a habit worth carrying over.)
    """

    def __init__(self, root: str, image_size: int, fit: str = "resize"):
        from torchvision import transforms

        self.files = sorted(
            p for p in Path(root).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not self.files:
            raise ValidationError(f"no images found in {root}")
        geom = ([transforms.Resize((image_size, image_size))] if fit == "resize"
                else [transforms.Resize(image_size), transforms.RandomCrop(image_size)])
        self.tf = transforms.Compose(geom + [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(self.files[i]) as im:
            return self.tf(im.convert("RGB"))


def _build_model(cfg: DiffusersTrainConfig):
    """Return ``(net, scheduler_config, trainable_params, note)``."""
    from app.core.engine import warmup

    warmup.wait_ready()  # diffusers' lazy imports are not safe beside the launch warm-up
    from diffusers import UNet2DModel

    from app.core.backends.xurdif import cosine_betas

    if cfg.mode == "scratch":
        spec = dict(SCRATCH_PRESETS[cfg.preset])
        spec.pop("label", None)
        arch = spec.pop("architecture", "unet2d")
        spec["sample_size"] = cfg.image_size

        if arch == "tinyunet":
            from .tinyunet import TinyUNet2DModel

            net = TinyUNet2DModel(**spec)
            # This architecture has only ever been trained on xurdif's cosine
            # schedule. Starting it on a linear one would work, but it would not
            # be the same model family any more, and nothing else in Kiln is set
            # up to expect that.
            sched = {"_class_name": "DDIMScheduler",
                     "num_train_timesteps": cfg.diffusion_steps,
                     "trained_betas": [float(b) for b in cosine_betas(cfg.diffusion_steps)],
                     "prediction_type": "epsilon"}
        else:
            net = UNet2DModel(in_channels=3, out_channels=3, norm_num_groups=32, **spec)
            sched = {"_class_name": "DDIMScheduler",
                     "num_train_timesteps": cfg.diffusion_steps,
                     "beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02,
                     "prediction_type": "epsilon"}
        return net, sched, list(net.parameters()), f"new {cfg.preset} model ({arch})"

    from app.core import backends

    from . import scheduler_config_of

    backend, ref = backends.resolve(cfg.base_model)
    if backend.name != "diffusers":
        raise ValidationError(
            f"'{cfg.base_model}' is a {backend.name} model; the Diffusers trainer "
            "cannot continue from one. Use the xurdif trainer for xurdif models.")
    adapter, meta = backend.load(ref, device="cpu")
    net = adapter.wrapped
    # The base model's own schedule, trained_betas included: continuing a
    # cosine model on linear noise would retrain it at the wrong noise levels.
    sched = scheduler_config_of(meta)

    if cfg.mode == "finetune":
        return net, sched, list(net.parameters()), f"fine-tuning {meta.name}"

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ValidationError(
            "LoRA needs the 'peft' package: pip install peft") from e

    # UNet2DModel is not a PeftAdapterMixin -- it has no add_adapter/
    # load_lora_adapter -- so the adapter is injected through PEFT directly
    # rather than through the diffusers LoRA API.
    net = get_peft_model(net, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        init_lora_weights="gaussian", target_modules=list(cfg.lora_targets),
    ))
    trainable = [p for p in net.parameters() if p.requires_grad]
    total = sum(p.numel() for p in net.parameters())
    pct = 100.0 * sum(p.numel() for p in trainable) / max(total, 1)
    return net, sched, trainable, f"LoRA r={cfg.lora_r} on {meta.name} ({pct:.2f}% trainable)"


def _noise_scheduler(sched_cfg: dict):
    """The scheduler that noises training batches.

    Built from the very dict ``_save_snapshot`` writes, through the same
    ``schedule_from_config`` the sampler uses to read it back. Reading the
    named-schedule keys here by hand once dropped ``trained_betas``: the tiny
    presets trained on linear noise and were then sampled on cosine.
    """
    from dataclasses import replace

    from diffusers import DDPMScheduler

    from . import schedule_from_config

    schedule = replace(schedule_from_config(sched_cfg),
                       prediction_type=sched_cfg.get("prediction_type", "epsilon"))
    return DDPMScheduler(**schedule.scheduler_kwargs())


def _save_snapshot(net, cfg: DiffusersTrainConfig, sched_cfg: dict, out_dir: Path,
                   milestone: int) -> Path:
    """Write ``model-N/`` as a self-contained, loadable model directory."""
    dest = out_dir / f"model-{milestone}"
    dest.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "lora":
        # Two artefacts: the adapter on its own (small, shareable, what LoRA is
        # for) and a merged copy so the snapshot is directly samplable like any
        # other model. Merging on a *copy* -- merge_and_unload is destructive.
        import copy

        net.save_pretrained(dest / "adapter")
        merged = copy.deepcopy(net).merge_and_unload()
        merged.save_pretrained(dest)
    else:
        target = getattr(net, "module", net)
        target.save_pretrained(dest)

    (dest / "scheduler_config.json").write_text(
        json.dumps(sched_cfg, indent=2), encoding="utf-8")

    # A model trained with the xurdif objective is low-contrast in x0, exactly
    # like one trained by the vendored engine, so it needs the same treatment on
    # the way to the screen. Record it with the model rather than inferring it
    # later from the architecture, which would be a proxy for the real reason.
    if cfg.objective == "xurdif":
        (dest / "kiln_provenance.json").write_text(json.dumps({
            "display": "xurdif",
            "objective": cfg.objective,
            "l1w": cfg.l1w,
            "ssimw": cfg.ssimw,
            "trained_by": "kiln-diffusers",
        }, indent=2), encoding="utf-8")
    return dest


def _render_sample(model_dir: Path, cfg: DiffusersTrainConfig, out_png: Path):
    """Render a preview from the snapshot that was just written.

    Deliberately samples the saved directory rather than the live training
    model: the thumbnail then shows exactly what the user gets when they load
    that checkpoint, and a preview that disagrees with the model is worse than
    no preview.

    Sampling during training is intentional and is not gated on the GPU being
    free -- watching a run take shape is the point of the Train screen.
    """
    from app.core.backends.hfdiffusers import loader as _loader
    from app.core.engine.sampler import SampleParams, sampler

    _loader.forget(str(model_dir))
    params = SampleParams(
        model_path=f"diffusers:{model_dir}",
        image_size=cfg.image_size, steps=20, sampler="unipc",
        seed=cfg.sample_seed if cfg.sample_seed >= 0 else None,
        postproc={},
    )
    last = None
    for frame in sampler.run(params):
        last = frame
    if last is not None:
        last["image"].save(out_png)
    return out_png if last is not None else None


def _device_overrides(cfg: DiffusersTrainConfig, device: str) -> list[str]:
    """Settle the settings this device cannot honour yet; one note per change.

    Apple's GPU trains in fp32 without torch.compile until mixed precision and
    the compiler have been tried there: neither has been measured on MPS, and a
    run that NaNs or fails to compile an hour in is worse than a slower one.
    Changed before run.json is written, so the run records what actually ran.
    """
    notes = []
    if device != "mps":
        return notes
    if cfg.precision != "no":
        notes.append(f"precision {cfg.precision} is not used on the Apple GPU yet; training in fp32")
        cfg.precision = "no"
    if cfg.compile_model:
        notes.append("torch.compile is not used on the Apple GPU yet; continuing without it")
        cfg.compile_model = False
    return notes


def _run(job: Job, cfg: DiffusersTrainConfig):
    import torch
    from torch.utils.data import DataLoader

    from app.core.devices import best_device
    from app.core.engine import lr_plan as lrplan
    from app.core.engine.trainer import (list_checkpoints, patch_run_meta,
                                         _rotate_run_log)

    device = best_device()
    device_notes = _device_overrides(cfg, device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = cfg.lr_plan or lrplan.compile_plan(
        None, lr=cfg.lr, train_steps=cfg.train_steps, save_every=cfg.save_every)
    meta = cfg.to_meta()
    meta["status"] = "training"
    meta["lr_schedule"] = plan.get("preset") or "custom"
    meta["lr_schedule_summary"] = lrplan.summarize(plan)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lrplan.write(out_dir, plan)
    job.detail["lr_plan"] = plan
    job.detail["lr"] = lrplan.lr_at(plan, 0)

    logf = None
    try:
        # Rotate rather than truncate, so continuing a run keeps the loss history
        # you need in order to see whether a rate change helped.
        logf = open(_rotate_run_log(out_dir), "w", encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    logbuf: list = job.detail.setdefault("log", [])

    def emit(line: str):
        if logf:
            logf.write(line + "\n")
            logf.flush()
        logbuf.append(line)
        if len(logbuf) > 500:
            del logbuf[: len(logbuf) - 500]
        job.detail["last_line"] = line

    try:
        from accelerate import Accelerator

        # Left alone, accelerate picks CUDA, then MPS, then the CPU -- the same
        # order as best_device, except that it cannot see KILN_DEVICE=cpu.
        accel = Accelerator(mixed_precision=cfg.precision,
                            gradient_accumulation_steps=cfg.accum,
                            cpu=device == "cpu")
        emit(f"device: {accel.device}  precision: {cfg.precision}")
        for line in device_notes:
            emit(line)
        job.detail["device"] = accel.device.type

        net, sched_cfg, trainable, note = _build_model(cfg)
        emit(note)
        job.detail["mode"] = cfg.mode
        job.detail["note"] = note

        if cfg.gradient_checkpointing:
            # Not every architecture implements it -- TinyUNet2DModel does not,
            # and it is cheap enough not to need it. Degrade rather than fail a
            # run over a memory optimisation.
            try:
                net.enable_gradient_checkpointing()
                emit("gradient checkpointing on")
            except Exception as e:  # noqa: BLE001
                emit(f"gradient checkpointing unavailable for this model: {e}")
        if cfg.compile_model:
            try:
                net = torch.compile(net)
                emit("torch.compile on")
            except Exception as e:  # noqa: BLE001
                emit(f"torch.compile unavailable, continuing without it: {e}")

        noise_sched = _noise_scheduler(sched_cfg)

        if cfg.manifest:
            from app.core.engine.train_data import ManifestDataset

            ds = ManifestDataset.from_snapshot(cfg.manifest, cfg.image_size, cfg.fit,
                                               engine="diffusers")
            emit(f"dataset: {len(ds.files)} images x {ds.variants} versions = {len(ds)} per pass")
        else:
            ds = ImageFolder(cfg.dataset, cfg.image_size, cfg.fit)
            emit(f"dataset: {len(ds)} images from {cfg.dataset}")
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, drop_last=len(ds) >= cfg.batch_size)
        opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        net, opt, dl = accel.prepare(net, opt, dl)
        lr_watch = lrplan.Watcher(out_dir, cfg.lr)
        # After prepare, so the average lives on the device the weights do.
        ema = _Ema(trainable, cfg.ema_decay) if cfg.use_ema else None
        if ema is not None:
            emit(f"snapshots save the averaged weights (EMA {cfg.ema_decay:g})")

        torch.manual_seed(cfg.seed)
        losses = job.detail.setdefault("losses", [])
        job.detail["checkpoints"] = []
        step = 0
        running = []
        micro = []
        t0 = time.time()

        while step < cfg.train_steps:
            for batch in dl:
                if job.cancelled() or step >= cfg.train_steps:
                    break
                # Evaluated on the pre-increment step, so "step N ran at lr(N)"
                # means the same here as in the vendored loop. Read at the top of
                # the iteration and so outside the pause barrier below: a run
                # paused while its plan is edited picks the change up on resume.
                rate_now, rate_changed = lr_watch.lr_for(step)
                for g in opt.param_groups:
                    g["lr"] = rate_now
                job.detail["lr"] = rate_now
                if rate_changed:
                    emit("lr %.6g from step %d" % (rate_now, step))
                    lrplan.append_event(out_dir, step, rate_now)
                    job.detail["lr_plan"] = lrplan.read(out_dir)
                    job.detail["lr_marks"] = lrplan.marks_for_chart(
                        lrplan.read_events(out_dir), job.detail["lr_plan"])
                with accel.accumulate(net):
                    clean = batch
                    noise = torch.randn_like(clean)
                    t = torch.randint(0, noise_sched.config.num_train_timesteps,
                                      (clean.shape[0],), device=clean.device).long()
                    noisy = noise_sched.add_noise(clean, noise, t)
                    pred = net(noisy, t, return_dict=False)[0]
                    loss, _parts = objectives.compute_loss(
                        pred, noisy, noise, clean, t,
                        noise_sched.alphas_cumprod,
                        prediction_type=sched_cfg.get("prediction_type", "epsilon"),
                        objective=cfg.objective, l1w=cfg.l1w, ssimw=cfg.ssimw,
                    )
                    accel.backward(loss)
                    if accel.sync_gradients:
                        accel.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    opt.zero_grad()

                # A step is one weight update, as in the vendored trainer: with
                # accumulation on, the batches before it only add to its gradient,
                # and train_steps, save_every and the rate plan all count updates.
                micro.append(float(loss.detach().item()))
                if not accel.sync_gradients:
                    continue
                step += 1
                if ema is not None:
                    ema.update()
                val = sum(micro) / len(micro)
                micro = []
                running.append(val)
                job.progress = min(step / max(cfg.train_steps, 1), 0.999)
                job.detail["step"] = step
                job.detail["loss"] = val
                if not job.paused():
                    rate = step / max(time.time() - t0, 1e-6)
                    job.message = (f"step {step} / {cfg.train_steps}  loss {val:.4f}"
                                   f"  ({rate:.1f} it/s)")
                # The xurdif trainer prints "<step>: <loss>" and load_run_view
                # parses that; matching it means one log format for both engines.
                if step % max(cfg.save_every // 2, 1) == 0:
                    emit(f"{step}: {val:.6f}")
                    losses.append({"step": step, "loss": val})

                if step % cfg.save_every == 0 or step == cfg.train_steps:
                    milestone = step // cfg.save_every
                    avg = sum(running) / max(len(running), 1)
                    running = []
                    emit(f"average loss: {avg:.6f}")
                    unwrapped = accel.unwrap_model(net)
                    with ema.applied() if ema is not None else nullcontext():
                        dest = _save_snapshot(unwrapped, cfg, sched_cfg, out_dir, milestone)
                    emit(f"saved {dest.name}")
                    try:
                        png = out_dir / f"sample-{milestone}.png"
                        if _render_sample(dest, cfg, png):
                            job.detail["sample"] = str(png)
                    except Exception as e:  # noqa: BLE001
                        emit(f"snapshot preview failed: {e}")
                    job.detail["checkpoints"] = list_checkpoints(out_dir, cfg.save_every)

                while job.paused() and not job.cancelled():
                    time.sleep(0.1)
            if job.cancelled():
                break

        job.detail["checkpoints"] = list_checkpoints(out_dir, cfg.save_every)
        if job.cancelled():
            job.status = "cancelled"
            job.message = "training stopped"
        elif job.status == "running":
            job.status = "done"
            job.progress = 1.0
            job.message = "training completed"
    except Exception as e:  # noqa: BLE001
        log.exception("diffusers training failed")
        emit(f"error: {e}")
        job.status = "error"
        job.message = str(e)
    finally:
        if logf:
            logf.close()
        try:
            patch_run_meta(out_dir, status=job.status, message=job.message,
                           step=job.detail.get("step"), finished_at=time.time())
        except Exception:  # noqa: BLE001
            pass


def start_training(cfg: DiffusersTrainConfig) -> Job:
    job = registry.create("train")
    job.message = "starting..."
    job.detail["out_dir"] = str(Path(cfg.out_dir))
    job.detail["log_path"] = str(Path(cfg.out_dir) / "train.log")
    job.detail["backend"] = "diffusers"
    job.detail["mode"] = cfg.mode
    t = threading.Thread(target=_run, args=(job, cfg), daemon=True)
    job.thread = t
    t.start()
    return job
