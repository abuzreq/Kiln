"""The Hugging Face Diffusers engine, behind the backend interface.

Scope is deliberately narrow: unconditional, pixel-space ``UNet2DModel``. Those
models predict the same thing xurdif's do, in the same space, so Kiln's denoise
loop drives them as-is. What differs is everything the app had previously
baked in as a constant -- the noise schedule, the display convention, the layer
names -- which is exactly what this class supplies.
"""
import importlib.util
from dataclasses import replace
from pathlib import Path

from app.core.backends.base import Backend, BetaSchedule, Capabilities, ModelRef

from app.core.backends.xurdif import graph as _xgraph

from . import graph, loader

# Top-level module names unique to the tinyunet architecture.
_TINY_ROOTS = {"init_conv", "downs", "ups", "final_conv",
               "mid_block1", "mid_attn", "mid_block2", "time_mlp", "time_emb"}


class DiffusersBackend(Backend):
    name = "diffusers"
    aliases = ("hf", "diffuser")
    capabilities = Capabilities(
        inference=True,
        train_from_scratch=True,
        finetune=True,
        # Verified: PEFT injects LoRA into UNet2DModel's attention projections
        # and resnet convs, and the adapter round-trips through save/load
        # exactly. diffusers' own LoRA API does not apply -- UNet2DModel is not
        # a PeftAdapterMixin -- so it goes through PEFT directly.
        lora=True,
        merge=True,
        bend=True,
        # Pixel-space only. A latent pipeline needs encode/decode around the
        # loop, which is why StableDiffusion* is refused rather than adapted.
        latent=False,
        train_devices=("cuda", "cpu"),
        infer_devices=("cuda", "cpu"),
    )

    def __init__(self):
        """Report only what this install can actually do.

        ``peft`` and ``accelerate`` are optional dependencies. Advertising LoRA
        on a machine without peft would put a button in the UI that fails when
        pressed, which is worse than not offering it -- so the capability is
        computed, not declared.
        """
        super().__init__()
        trains = _installed("accelerate")
        self.capabilities = replace(
            type(self).capabilities,
            train_from_scratch=trains,
            finetune=trains,
            lora=trains and _installed("peft"),
        )

    # --- identity -----------------------------------------------------
    def claims(self, locator: str) -> bool:
        """Only claim things we can positively identify.

        Never claim a bare string that merely *looks* like a repo id: a typo'd
        path would then be reported as an unsupported Hub model instead of a
        missing file. Explicit ``diffusers:owner/repo`` refs bypass this and go
        straight to the loader, which is where a repo id belongs.
        """
        try:
            return loader.is_local_model_dir(Path(locator))
        except Exception:  # noqa: BLE001
            return False

    # --- discovery ----------------------------------------------------
    def scan(self, sources):
        return loader.scan(sources)

    def describe(self, ref: ModelRef):
        return loader.describe(ref.locator, ref.revision)

    def probe(self, ref: ModelRef) -> dict:
        """Is this loadable, and if not, why? Reads metadata only."""
        return loader.probe(ref.locator, ref.revision)

    # --- loading ------------------------------------------------------
    def load(self, ref: ModelRef, device: str = "cpu", ema: bool = True):
        return loader.load_net(ref.locator, ref.revision, device=device, ema=ema)

    # --- sampling -----------------------------------------------------
    def schedule(self, ref: ModelRef, train_steps: int | None = None) -> BetaSchedule:
        """The model's own schedule, from its ``scheduler_config.json``.

        ``train_steps`` from the caller is ignored on purpose. For xurdif it is
        a real user choice; here the value is a property of the trained weights,
        and overriding it is how you get a plausible-looking but wrong image.
        """
        import numpy as np

        meta = self.describe(ref)
        x = meta.extra
        trained = x.get("trained_betas")
        return BetaSchedule(
            num_train_timesteps=int(x.get("num_train_timesteps", 1000)),
            trained_betas=np.asarray(trained, dtype=np.float32) if trained else None,
            beta_schedule=x.get("beta_schedule", "linear"),
            beta_start=x.get("beta_start", 0.0001),
            beta_end=x.get("beta_end", 0.02),
            # Always epsilon, whatever the model itself predicts. The denoise
            # loop converts the model's output to eps *before* calling
            # scheduler.step (see sampler.Sampler.run), so the scheduler is only
            # ever handed noise. Deriving this from the model's own
            # prediction_type made an x0-predicting model take the wrong step and
            # silently produced a different image -- caught by the conversion
            # parity test, which samples the same weights through both backends.
            prediction_type="epsilon",
        )

    def to_display(self, x0, meta=None):
        """Straight [-1,1] -> [0,1] by default, with one documented exception.

        A model trained the ordinary way already occupies the range it means to;
        measurement showed xurdif's fixed-std stretch applying a ~1.6x contrast
        boost to a correct sample. But a *converted* xurdif checkpoint is still
        a xurdif model -- its raw x0 really is low-contrast -- so it carries
        ``display: "xurdif"`` in its provenance and keeps the original
        treatment. Without this, converting a model would visibly change it.
        """
        import torch

        if meta is not None and (meta.extra or {}).get("display") == "xurdif":
            from app.core.backends.xurdif import XurdifBackend

            return XurdifBackend().to_display(x0)
        im = (x0.clone().clamp(-1, 1) + 1) * 0.5
        im = torch.nan_to_num(im, nan=0.0, posinf=1.0, neginf=0.0)
        return im.clamp(0, 1)

    # --- craft --------------------------------------------------------
    # The layer vocabulary belongs to the *architecture*, not the engine: this
    # backend can load two of them. TinyUNet2DModel is xurdif's network, so it
    # keeps xurdif's names and reuses that (already tested) graph module rather
    # than duplicating it.
    attention_types = tuple(set(graph.ATTENTION_TYPES) | set(_xgraph.ATTENTION_TYPES))
    block_types = tuple(set(graph.BLOCK_TYPES) | set(_xgraph.BLOCK_TYPES))

    @staticmethod
    def _vocab_for_net(net):
        inner = getattr(net, "wrapped", net)
        # The two namespaces are disjoint: tinyunet has `downs`, UNet2DModel has
        # `down_blocks`. Duck-typing beats threading a class name through every
        # call site that only ever has the module in hand.
        return _xgraph if hasattr(inner, "downs") else graph

    @staticmethod
    def _vocab_for_key(key: str):
        return _xgraph if key.split(".", 1)[0] in _TINY_ROOTS else graph

    def layer_graph(self, net, image_size: int = 64) -> dict:
        inner = getattr(net, "wrapped", net)
        return self._vocab_for_net(net).layer_graph(inner, image_size=image_size)

    def bend_points(self, net) -> list[str]:
        inner = getattr(net, "wrapped", net)
        return self._vocab_for_net(net)._ordered_points(inner)

    def stage_of_point(self, name: str) -> str:
        return self._vocab_for_key(name)._stage_of(name)

    def stage_of_key(self, key: str) -> str:
        return self._vocab_for_key(key).stage_of_key(key)

    # --- training -----------------------------------------------------
    training_modes = ("scratch", "finetune", "lora")

    def training_presets(self) -> dict:
        from . import training

        return training.presets()

    def training_config(self, body: dict, dataset, out_dir):
        from . import training

        return training.config_from_body(body, dataset, out_dir)

    def start_training(self, cfg):
        from . import training

        return training.start_training(cfg)

    # --- merging ------------------------------------------------------
    def merge_slots(self, ref: ModelRef) -> dict:
        """One slot: a Diffusers repo publishes a single set of weights.

        There is no ema/model pair to keep in step -- repos that shipped an EMA
        variant did it as a separate repo entirely.
        """
        net, _ = loader.load_net(ref.locator, ref.revision, device="cpu")
        return {"unet": net.wrapped.state_dict()}

    def write_merged(self, ref: ModelRef, slots: dict, dest_dir, out_name: str,
                     info: dict) -> str:
        """Write a real Diffusers model directory, not a bag of tensors.

        Saving via ``save_pretrained`` means the result is loadable by anything
        in the ecosystem -- and, more immediately, by this backend's own scan,
        so a merge shows up in the model list like any other model.
        """
        import json

        from diffusers import UNet2DModel

        from .tinyunet import TinyUNet2DModel

        meta = self.describe(ref)
        dest = Path(dest_dir) / out_name
        dest.mkdir(parents=True, exist_ok=True)

        cls = (TinyUNet2DModel if meta.extra.get("model_class") == loader.TINYUNET_CLASS
               else UNet2DModel)
        net = cls.from_config(
            cls.load_config(
                ref.locator, subfolder=meta.extra.get("unet_subfolder"),
                revision=ref.revision, local_files_only=loader.offline(),
            )
        )
        net.load_state_dict(slots["unet"])
        net.save_pretrained(dest)

        # Carry the schedule across: without scheduler_config.json the merged
        # model would silently fall back to defaults, which is the same class of
        # bug as using the wrong betas in the first place.
        sched = {k: meta.extra[k] for k in
                 ("num_train_timesteps", "beta_schedule", "beta_start", "beta_end")}
        if meta.extra.get("trained_betas"):
            # A converted xurdif model pins its exact curve; naming a schedule
            # instead would quietly resample it on a different one.
            sched["trained_betas"] = list(meta.extra["trained_betas"])
        sched["_class_name"] = "DDIMScheduler"
        sched["prediction_type"] = "epsilon" if meta.pred == "eps" else "sample"
        (dest / "scheduler_config.json").write_text(
            json.dumps(sched, indent=2), encoding="utf-8")
        (dest / "kiln_merge.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        loader.forget(str(dest))
        return str(dest)


def _installed(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


__all__ = ["DiffusersBackend", "graph", "loader"]
