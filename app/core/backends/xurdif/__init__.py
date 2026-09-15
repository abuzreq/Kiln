"""The vendored xurdif engine, behind the backend interface.

Nothing here is new behaviour: the schedule, the display convention and the
layer vocabulary are the constants Kiln has always used, relocated so a second
backend can hold different ones. Any change to this file changes what existing
models produce -- ``scripts/smoke_golden.py`` is the gate.
"""
import math
from pathlib import Path

from app.core.backends.base import Backend, BetaSchedule, Capabilities, ModelRef

from . import attn as attn_spec
from . import graph, loader


def cosine_betas(timesteps: int):
    """xurdif's beta schedule.

    Hardcoded identically in ``GaussianDiffusion`` and ``DDIMDiffusion``
    (vendor/xurdif/xurdif.py), so every xurdif checkpoint in existence was
    trained against it. Reproduced here rather than imported because the vendored
    versions build the tensor straight onto CUDA.
    """
    import torch

    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 0, 0.999).numpy()


def _clean_mults(raw, image_size: int) -> list:
    """Channel multipliers, validated before they can reach the subprocess.

    This is the one field the trainer took on trust. A bad value did not fail
    here -- it failed several seconds later inside the training subprocess, as a
    shape mismatch on the UNet's skip concatenation, surfacing to the user as a
    stack trace in the log panel. Each entry halves the feature map and the skips
    concatenate, so the image size has to divide by ``2 ** len(mults)``.
    """
    from app.core.engine.sampler import align_size
    from utils.exceptions import ValidationError

    mults = raw or [1, 2, 2, 2]
    if isinstance(mults, str):
        mults = [x for x in (part.strip() for part in mults.split(",")) if x]
    try:
        mults = [int(m) for m in mults]
    except (TypeError, ValueError):
        raise ValidationError(
            "channel multipliers must be whole numbers, e.g. 1,2,2,2")
    if not 1 <= len(mults) <= 8 or any(m < 1 for m in mults):
        raise ValidationError(
            "channel multipliers must be 1-8 positive whole numbers, e.g. 1,2,2,2")
    step = 2 ** len(mults)
    if image_size % step:
        raise ValidationError(
            f"image size {image_size} must divide by {step} for {len(mults)} "
            f"channel multipliers - try {align_size(image_size, mults)}")
    return mults


class XurdifBackend(Backend):
    name = "xurdif"
    aliases = ("xur",)
    capabilities = Capabilities(
        inference=True,
        train_from_scratch=True,
        finetune=True,
        # The nets are ~3.8M params and their LoRA-able leaves have positional
        # names (`0`, `1`, `conv`) from living inside nn.Sequential. Full
        # fine-tuning is already cheap here, so adapters would buy nothing.
        lora=False,
        merge=True,
        bend=True,
        latent=False,
        # xurdiftrainer.py hardcodes .cuda(); there is no CPU training path.
        train_devices=("cuda",),
        infer_devices=("cuda", "cpu"),
    )

    # --- identity -----------------------------------------------------
    def claims(self, locator: str) -> bool:
        try:
            p = Path(locator)
        except OSError:
            return False
        return p.suffix == ".pt"

    # --- discovery ----------------------------------------------------
    def scan(self, sources):
        return loader.scan(sources)

    def describe(self, ref: ModelRef):
        return loader.describe(ref.locator)

    # --- loading ------------------------------------------------------
    def load(self, ref: ModelRef, device: str = "cpu", ema: bool = True):
        return loader.load_net(ref.locator, device=device, ema=ema)

    # --- sampling -----------------------------------------------------
    def schedule(self, ref: ModelRef, train_steps: int | None = None) -> BetaSchedule:
        """xurdif does not record its schedule length in the checkpoint.

        ``--steps`` at training time and ``train_steps`` at sampling time are
        separate values that the user is expected to keep in step; the sampler
        has always passed its own, so honour it and default the same way.
        """
        n = int(train_steps or 1000)
        return BetaSchedule(
            num_train_timesteps=n,
            trained_betas=cosine_betas(n),
            prediction_type="epsilon",
        )

    def to_display(self, x0, meta=None):
        """Per-sample contrast normalisation, matching xurdifapp3.

        Forces every output to std 0.18. Correct for these models -- their raw
        x0 is low-contrast -- but it is a convention, not a colour-space fact,
        so it must not follow the tensor into another backend.
        """
        import torch

        im = (x0.clone().clamp(-1, 1) + 1) * 0.5
        im = torch.nan_to_num(im, nan=0.0, posinf=1.0, neginf=0.0)
        m = im.mean(dim=(1, 2, 3), keepdim=True)
        sd = im.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        im = (im - m) * (0.18 / sd) + 0.5
        return im.clamp(0, 1)

    # --- craft --------------------------------------------------------
    attention_types = graph.ATTENTION_TYPES
    block_types = graph.BLOCK_TYPES

    def layer_graph(self, net, image_size: int = 64) -> dict:
        return graph.layer_graph(net, image_size=image_size)

    def bend_points(self, net) -> list[str]:
        return graph._ordered_points(net)

    def stage_of_point(self, name: str) -> str:
        return graph._stage_of(name)

    def stage_of_key(self, key: str) -> str:
        return graph.stage_of_key(key)

    # --- merging ------------------------------------------------------
    def merge_slots(self, ref: ModelRef) -> dict:
        import torch

        data = torch.load(ref.locator, map_location="cpu", weights_only=False)
        return {k: data[k] for k in ("model", "ema") if k in data}

    def write_merged(self, ref: ModelRef, slots: dict, dest_dir, out_name: str,
                     info: dict) -> str:
        import torch

        meta = self.describe(ref)
        out = {
            "step": 0,
            "mults": meta.mults,
            "mtype": meta.mtype,
            "pred": meta.pred,
            **info,
            **slots,
        }
        # Recorded the way upstream's trainer records it: ``attn_conf`` plus an
        # ``opt`` namespace. A merge has no training options of its own, so its
        # ``opt`` names only the architecture -- enough for upstream's
        # generation script, which rebuilds the net from ``opt.attn_config``.
        # For a conf model the layout decides which tensors exist, and
        # check_compat has already made sure both parents agree on it.
        import argparse

        out["attn_conf"] = meta.attn
        out["opt"] = argparse.Namespace(
            model=meta.mtype, mults=list(meta.mults), pred=meta.pred,
            attn=meta.attn, attn_config=attn_spec.parse(meta.attn),
        )
        # Both slots must exist: everything downstream picks one by name, and a
        # checkpoint missing "ema" silently loads the non-averaged weights.
        if "model" not in out and "ema" in out:
            out["model"] = out["ema"]
        if "ema" not in out and "model" in out:
            out["ema"] = out["model"]
        dest = Path(dest_dir) / f"{out_name}.pt"
        torch.save(out, str(dest))
        return str(dest)

    # --- training -----------------------------------------------------
    # "continue" is xurdif's own resume: same architecture, all weights, via
    # the trainer's --load. It is not a Diffusers-style fine-tune and is
    # deliberately named differently so the two are not confused.
    training_modes = ("scratch", "continue")

    def training_presets(self) -> dict:
        from app.backend.routes.train import CONFIG_PRESETS

        return CONFIG_PRESETS

    def training_config(self, body: dict, dataset, out_dir):
        """Build a TrainConfig. Moved verbatim from routes/train.py."""
        from app.core.engine.trainer import TrainConfig
        from utils.validators import as_float, as_int

        resume = body.get("resume") or None
        if resume and not Path(resume).exists():
            from utils.exceptions import NotFoundError

            raise NotFoundError("starting checkpoint not found")
        image_size = as_int(body.get("image_size", 512), "image_size", 32, 4096)
        mtype = body.get("mtype") or attn_spec.MTYPE
        mults = _clean_mults(body.get("mults"), image_size)
        return TrainConfig(
            dataset=str(dataset),
            out_dir=str(out_dir),
            name=body.get("model_name", Path(str(out_dir)).name),
            image_size=image_size,
            batch_size=as_int(body.get("batch_size", 8), "batch_size", 1, 64),
            diffusion_steps=as_int(body.get("diffusion_steps", 1000), "diffusion_steps", 10, 4000),
            train_steps=as_int(body.get("train_steps", 280000), "train_steps", 100, 5_000_000),
            accum=as_int(body.get("accum", 10), "accum", 1, 128),
            lr=as_float(body.get("lr", 4e-4), "lr", 1e-6, 1.0),
            loss_type=body.get("loss_type", "l1"),
            l1w=as_float(body.get("l1w", 1.0), "l1w", 0, 100),
            ssimw=as_float(body.get("ssimw", 0.0), "ssimw", 0, 100),
            pred=body.get("pred", "x0"),
            mtype=mtype,
            mults=mults,
            attn=attn_spec.validate(body.get("attn"), mults) if mtype == attn_spec.MTYPE else None,
            fit=body.get("fit", "resize"),
            nsamples=as_int(body.get("nsamples", 1), "nsamples", 1, 16),
            sample_seed=as_int(body.get("sample_seed", 42), "sample_seed", -1, 2 ** 31 - 1),
            save_every=as_int(body.get("save_every", 1000), "save_every", 10, 100000),
            amp=bool(body.get("amp", False)),
            resume=resume,
            nostrict=bool(body.get("nostrict", False)),
            edge_loss=bool(body.get("edge_loss", True)),
        )

    def start_training(self, cfg):
        from app.core.engine.trainer import start_training

        return start_training(cfg)


__all__ = ["XurdifBackend", "cosine_betas", "graph", "loader"]
