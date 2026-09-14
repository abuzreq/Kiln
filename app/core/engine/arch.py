"""Reconstruct a xurdif UNet from checkpoint metadata.

A xurdif checkpoint records its architecture as ``mtype`` (model name) and
``mults`` (channel multipliers). This module maps ``mtype`` to the matching
class in the vendored ``alt_models`` package and builds the network.
"""
import importlib

import torch.nn as nn

from ._vendor import ensure_on_path

# mtype -> (module, class_name). Only architectures shipped in the vendored
# snapshot are guaranteed importable; others raise a clear error when requested.
MODEL_REGISTRY = {
    "tinyunet_conf_attention": ("alt_models.tinyunet_conf_attn", "TinyUNetWithAttn"),
    "tinyunet_with_attention3": ("alt_models.tinyunet_with_attn3", "TinyUNetWithAttn"),
    "tinyunet_with_attention": ("alt_models.tinyunet_with_attn", "TinyUNetWithAttn"),
    "tinyunet": ("alt_models.tinyunet", "TinyUNet"),
    "unet0": ("alt_models.Unet0", "Unet"),
    "unet0k5": ("alt_models.Unet0k5", "Unet"),
    "unet1": ("alt_models.Unet1", "Unet"),
    "unet2": ("alt_models.Unet2", "Unet"),
    "unetcn0": ("alt_models.UnetCN0", "Unet"),
    "unet2d": ("alt_models.Unet2d", "Unet"),
}

# The architecture assumed for a checkpoint that records none. This is the
# oldest kind of file Kiln reads and must stay on the class those files were
# trained with; the default for *new* runs lives in the trainer and is a
# different thing.
DEFAULT_MTYPE = "tinyunet_with_attention3"

# The one architecture whose shape is not fixed by (mtype, mults) alone.
CONF_MTYPE = "tinyunet_conf_attention"


def available_architectures() -> list[str]:
    """Return the mtypes whose modules can actually be imported here."""
    ensure_on_path()
    out = []
    for mtype, (mod, _cls) in MODEL_REGISTRY.items():
        try:
            importlib.import_module(mod)
            out.append(mtype)
        except Exception:  # noqa: BLE001
            continue
    return out


def build_unet(mtype: str, mults, dim: int = 64, channels: int = 3,
               attn_config: dict | None = None) -> nn.Module:
    """Build the bare UNet.

    ``attn_config`` is the parsed attention layout and only means something to
    ``tinyunet_conf_attention``; ``None`` there is the constructor's own default
    (full attention at the bottleneck). The other classes do not take it.
    """
    ensure_on_path()
    if mtype not in MODEL_REGISTRY:
        raise ValueError(f"unknown model type: {mtype}")
    mod_name, cls_name = MODEL_REGISTRY[mtype]
    module = importlib.import_module(mod_name)
    cls = getattr(module, cls_name)
    if mtype == CONF_MTYPE:
        return cls(dim=dim, dim_mults=tuple(mults), attn_config=attn_config)
    return cls(dim=dim, dim_mults=tuple(mults))
