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

DEFAULT_MTYPE = "tinyunet_with_attention3"


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


def build_unet(mtype: str, mults, dim: int = 64, channels: int = 3) -> nn.Module:
    ensure_on_path()
    if mtype not in MODEL_REGISTRY:
        raise ValueError(f"unknown model type: {mtype}")
    mod_name, cls_name = MODEL_REGISTRY[mtype]
    module = importlib.import_module(mod_name)
    cls = getattr(module, cls_name)
    return cls(dim=dim, dim_mults=tuple(mults))
