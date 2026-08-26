"""Convert a xurdif ``.pt`` checkpoint into a Diffusers model directory.

This is the one conversion in Kiln that is actually lossless, and it is lossless
for a specific reason: the network on both sides is the same network. Converting
to ``UNet2DModel`` would be impossible (72 tensors vs 190, no shared keys, no
structural mapping); converting to ``TinyUNet2DModel`` is a re-serialisation.

The part that is easy to get wrong is the **schedule**. Every xurdif checkpoint
was trained against xurdif's own cosine betas, which are *not* the same curve as
diffusers' ``squaredcos_cap_v2`` -- measured max difference 0.249. Recording the
schedule by name would therefore silently change what a converted model
produces, so the exact betas are written into the scheduler config instead.
"""
import json
from pathlib import Path

from utils.exceptions import NotFoundError, ValidationError
from utils.logger import get_logger

from .tinyunet import SOURCE_MTYPE, TinyUNet2DModel

log = get_logger("diffusers.convert")

DEFAULT_TRAIN_TIMESTEPS = 1000


def _run_meta_for(pt: Path) -> dict:
    """The run.json beside a training snapshot, if this came from one."""
    f = pt.parent / "run.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def resolve_train_timesteps(pt: Path, override=None) -> tuple[int, str]:
    """Work out the diffusion schedule length, and say where it came from.

    xurdif does not store this in the checkpoint -- ``--steps`` at training time
    and ``train_steps`` at sampling time are separate values a user is expected
    to keep in step. Getting it wrong rescales the whole noise schedule, so the
    source is reported rather than silently defaulted.
    """
    if override:
        return int(override), "caller"
    meta = _run_meta_for(pt)
    if meta.get("diffusion_steps"):
        return int(meta["diffusion_steps"]), "run.json"
    try:
        from app.core import library

        card = library.read_card(pt) or {}
        if card.get("diffusion_steps"):
            return int(card["diffusion_steps"]), "model card"
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_TRAIN_TIMESTEPS, "default"


def _sample_size_for(pt: Path) -> int | None:
    meta = _run_meta_for(pt)
    if meta.get("image_size"):
        return int(meta["image_size"])
    try:
        from app.core import library

        card = library.read_card(pt) or {}
        if card.get("image_size"):
            return int(card["image_size"])
    except Exception:  # noqa: BLE001
        pass
    return None


def convert_checkpoint(pt_path, dest_dir, *, ema: bool = True,
                       num_train_timesteps=None, overwrite: bool = False) -> dict:
    """Write ``pt_path`` out as a Diffusers model directory. Returns a report.

    ``ema`` picks which of the checkpoint's two weight sets to convert. They are
    different models -- the EMA copy is the smoother one and is what Kiln samples
    by default -- so the choice is recorded in the output.
    """
    from app.core.backends.xurdif import cosine_betas, loader as xloader

    pt = Path(pt_path)
    if not pt.exists():
        raise NotFoundError(f"checkpoint not found: {pt}")
    dest = Path(dest_dir)
    if dest.exists() and any(dest.iterdir()) and not overwrite:
        raise ValidationError(f"'{dest}' already exists and is not empty")

    meta = xloader.describe(pt)
    if meta.mtype != SOURCE_MTYPE:
        raise ValidationError(
            f"only '{SOURCE_MTYPE}' can be re-homed today; this checkpoint is "
            f"'{meta.mtype}'. The other vendored architectures are not shipped "
            "in the snapshot, so there is nothing to convert against.")

    net, _ = xloader.load_net(str(pt), device="cpu", ema=ema)
    sample_size = _sample_size_for(pt)
    model = TinyUNet2DModel.from_vendored(net, sample_size=sample_size)

    timesteps, source = resolve_train_timesteps(pt, num_train_timesteps)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)

    # The exact curve, not its name -- see the module docstring.
    betas = cosine_betas(timesteps)
    scheduler = {
        "_class_name": "DDIMScheduler",
        "num_train_timesteps": int(timesteps),
        "prediction_type": "epsilon" if meta.pred == "eps" else "sample",
        "trained_betas": [float(b) for b in betas],
        "clip_sample": False,
    }
    (dest / "scheduler_config.json").write_text(
        json.dumps(scheduler, indent=2), encoding="utf-8")

    provenance = {
        "converted_from": str(pt),
        "source_name": pt.stem,
        "source_mtype": meta.mtype,
        "source_step": meta.step,
        "weights": "ema" if ema else "model",
        "mults": list(meta.mults),
        "pred": meta.pred,
        "num_train_timesteps": int(timesteps),
        "num_train_timesteps_source": source,
        "sample_size": sample_size,
        # These models are low-contrast in x0 and Kiln has always stretched them
        # to a fixed std on the way to the screen. Record it so the conversion
        # does not silently change how the model looks.
        "display": "xurdif",
    }
    (dest / "kiln_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    from . import loader as dloader

    dloader.forget(str(dest))
    log.info("converted %s -> %s (T=%d from %s)", pt.name, dest, timesteps, source)
    return {"path": str(dest), "name": dest.name, **provenance}
