"""Contracts every model backend implements.

Kiln grew up around one engine (the vendored xurdif snapshot), so a handful of
its assumptions were written into the app as constants rather than as questions
it could ask: the beta schedule, the display convention, and the UNet's layer
names. A second backend cannot share those, so each one becomes a method here.

The rest of the app is expected to ask *capability* questions -- "can this model
be bent?", "does this backend do LoRA?" -- and never to branch on a backend name.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Capabilities:
    """What a backend can do. Static per backend, not per model."""

    inference: bool = False
    train_from_scratch: bool = False
    finetune: bool = False
    lora: bool = False
    merge: bool = False
    bend: bool = False
    # False for every backend today. This is the seam for a future latent /
    # text-conditioned backend: the sampler works in pixel space, so anything
    # that sets this true also needs an encode/decode step around the loop.
    latent: bool = False
    # Devices the backend can *train* on. xurdif's trainer hardcodes .cuda().
    # Enforced by app.core.devices.train_block_reason.
    train_devices: tuple[str, ...] = ("cuda",)
    infer_devices: tuple[str, ...] = ("cuda", "mps", "cpu")

    def to_dict(self) -> dict:
        return {
            "inference": self.inference,
            "train_from_scratch": self.train_from_scratch,
            "finetune": self.finetune,
            "lora": self.lora,
            "merge": self.merge,
            "bend": self.bend,
            "latent": self.latent,
            "train_devices": list(self.train_devices),
            "infer_devices": list(self.infer_devices),
        }


@dataclass(frozen=True)
class ModelRef:
    """Where a model lives, and which backend owns it.

    Critically, ``str(ref)`` round-trips to the string the app already stores.
    Every PNG recipe card, star entry, library preset and API payload keeps
    ``model_path`` as a plain string, so a xurdif ref must serialise back to its
    bare filesystem path -- byte-identical to what is on disk today. Without
    that, every existing capture and recipe would need migrating.
    """

    backend: str
    locator: str
    revision: str | None = None

    def __str__(self) -> str:
        if self.backend == "xurdif":
            return self.locator                     # unchanged from today
        base = f"{self.backend}:{self.locator}"
        return f"{base}@{self.revision}" if self.revision else base

    @property
    def is_path(self) -> bool:
        try:
            return Path(self.locator).exists()
        except OSError:
            return False


@dataclass
class ModelDescriptor:
    """Everything the app needs to list, describe and validate one model.

    The first block of fields is exactly what ``ModelMeta`` exposed, kept so the
    frontend and the existing API responses do not change. ``backend`` and
    ``size_multiple`` are additive.
    """

    path: str
    name: str
    mtype: str
    mults: list
    pred: str
    step: int | None = None
    size_mb: float = 0.0
    source: str = "workspace"
    thumbnail: str | None = None
    # --- additive ---------------------------------------------------
    backend: str = "xurdif"
    # Generalises ``2 ** len(mults)``: the multiple every image dimension must
    # divide by for the UNet's skip connections to concatenate.
    size_multiple: int = 16
    # Whether asking for EMA weights can actually change anything:
    #   "distinct" -- the model carries a separate, different averaged copy
    #   "same"     -- it carries one, but identical to the raw weights
    #   "none"     -- there is only one set of weights
    # Kiln has always offered an "Use EMA weights" toggle and silently fallen
    # back when there was nothing to fall back to; this is what lets the UI say
    # so instead.
    ema: str = "none"
    # Resolution the model was trained at, when it records one. Diffusers
    # configs carry it; xurdif checkpoints do not, so it stays None there and
    # the UI simply says nothing rather than guessing.
    sample_size: int | None = None
    # Attention layout, canonical spec string ("-1:linear,mid:full"), for the
    # one xurdif architecture whose shape is not fixed by (mtype, mults). None
    # for every other model, and for a conf checkpoint that never recorded it.
    attn: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "mtype": self.mtype,
            "mults": self.mults,
            "pred": self.pred,
            "step": self.step,
            "size_mb": round(self.size_mb, 2),
            "source": self.source,
            "thumbnail": self.thumbnail,
            "backend": self.backend,
            "size_multiple": self.size_multiple,
            "ema": self.ema,
            "sample_size": self.sample_size,
            "attn": self.attn,
        }

    def copy(self) -> "ModelDescriptor":
        return replace(self)


@dataclass(frozen=True)
class BetaSchedule:
    """How a backend's models were noised during training.

    This exists because it is the one difference that fails *silently*. xurdif
    hardcodes a cosine schedule; Hugging Face DDPM checkpoints are usually
    linear. Feeding a linear-trained model the cosine alphas tells it it is at a
    ~6x different noise level -- measured at t=500: 0.493 vs 0.078 -- and it
    produces a degraded image rather than an error.
    """

    num_train_timesteps: int
    trained_betas: np.ndarray | None = None
    beta_schedule: str | None = None
    beta_start: float | None = None
    beta_end: float | None = None
    prediction_type: str = "epsilon"

    def scheduler_kwargs(self) -> dict:
        """Constructor kwargs for any diffusers scheduler class."""
        kw: dict = {
            "num_train_timesteps": int(self.num_train_timesteps),
            "prediction_type": self.prediction_type,
        }
        if self.trained_betas is not None:
            kw["trained_betas"] = self.trained_betas
        else:
            if self.beta_schedule:
                kw["beta_schedule"] = self.beta_schedule
            if self.beta_start is not None:
                kw["beta_start"] = self.beta_start
            if self.beta_end is not None:
                kw["beta_end"] = self.beta_end
        return kw


class Backend:
    """One model engine. Subclasses fill in what their models actually do.

    ``load`` has the tightest contract: whatever module it returns must accept
    ``(x, t) -> Tensor`` the way xurdif's UNets do, because the denoise loop,
    the bending hooks and the introspection probe are all written against that
    signature. Adapting a foreign forward signature is the backend's job.
    """

    name: str = "base"
    aliases: tuple[str, ...] = ()
    capabilities = Capabilities()

    # --- identity -----------------------------------------------------
    def claims(self, locator: str) -> bool:
        """Does this backend recognise ``locator`` as one of its models?"""
        raise NotImplementedError

    # --- discovery ----------------------------------------------------
    def scan(self, sources: list[tuple[Path, str]]) -> list[ModelDescriptor]:
        raise NotImplementedError

    def iter_scan(self, sources: list[tuple[Path, str]], skip: "set[str] | None" = None):
        """``scan``, one model at a time, for callers that show them as they come.

        Reading a checkpoint is the slow part of a scan, so a backend that can
        yield between reads should override this; the default waits for all.
        ``skip`` holds paths the caller already knows, which need not be read.
        """
        yield from (m for m in self.scan(sources) if not skip or m.path not in skip)

    def describe(self, ref: ModelRef) -> ModelDescriptor:
        raise NotImplementedError

    # --- loading ------------------------------------------------------
    def load(self, ref: ModelRef, device: str = "cpu", ema: bool = True):
        """Return ``(net, descriptor)``; ``net(x, t)`` must yield a Tensor."""
        raise NotImplementedError

    # --- sampling -----------------------------------------------------
    def schedule(self, ref: ModelRef, train_steps: int | None = None) -> BetaSchedule:
        raise NotImplementedError

    def to_display(self, x0, meta: "ModelDescriptor | None" = None):
        """Map a model's x0 prediction into a [0,1] tensor for display.

        ``meta`` matters because the convention belongs to the *model*, not the
        engine: a backend can hold models trained to different output ranges, and
        a xurdif model converted into Diffusers format still wants xurdif's
        contrast stretch.
        """
        raise NotImplementedError

    # --- craft --------------------------------------------------------
    #: Module class names behind Craft's "attention" and "blocks" group chips.
    attention_types: tuple[str, ...] = ()
    block_types: tuple[str, ...] = ()

    def layer_graph(self, net, image_size: int = 64) -> dict:
        raise NotImplementedError

    def bend_points(self, net) -> list[str]:
        """Module names that emit a feature map, in forward order."""
        raise NotImplementedError

    def stage_of_point(self, name: str) -> str:
        """Bucket one module name into encoder / mid / decoder / other."""
        raise NotImplementedError

    def slice_points(self, net) -> dict[str, dict]:
        """Targets that bend a channel range of a module's input (``hook:
        "pre"``) or output (``"out"``) instead of a whole module output: skip
        connections, fused q/k/v. ``{id: {module, hook, start, stop}}``; a
        backend with none returns ``{}``."""
        return {}

    def stage_of_key(self, key: str) -> str:
        """Bucket one state-dict key into encoder / mid / decoder / other."""
        raise NotImplementedError

    # --- merging ------------------------------------------------------
    # Blending weights is shared arithmetic; only reading and writing the
    # checkpoint differs, so that is all a backend has to supply.
    def merge_slots(self, ref: "ModelRef") -> dict:
        """Named weight sets to blend, e.g. ``{"model": sd, "ema": sd}``."""
        raise NotImplementedError

    def write_merged(self, ref: "ModelRef", slots: dict, dest_dir, out_name: str,
                     info: dict) -> str:
        """Persist blended slots in this backend's own format; return the path."""
        raise NotImplementedError

    # --- training -----------------------------------------------------
    #: Training modes this backend understands, e.g. ("scratch", "finetune").
    training_modes: tuple[str, ...] = ()

    def supports_training_mode(self, mode: str) -> bool:
        return mode in self.training_modes

    def training_presets(self) -> dict:
        """Named starting configurations for the Train screen."""
        return {}

    def training_config(self, body: dict, dataset, out_dir):
        """Validate a request into this backend's own training config."""
        raise NotImplementedError

    def start_training(self, cfg):
        """Begin a run and return a Job the UI can poll."""
        raise NotImplementedError
