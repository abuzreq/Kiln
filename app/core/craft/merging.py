"""Two-way model merging for xurdif checkpoints.

Strategies:
- linear:    out = (1-a)*A + a*B, per tensor
- slerp:     spherical interpolation per tensor (preserves norm/character better)
- blockwise: linear, but with a different alpha per stage (encoder / mid / decoder)

Guardrails: both checkpoints must share architecture (mtype) and channel
multipliers (mults); tensor keys and shapes must match. Merges are strictly 2-way.
"""
from pathlib import Path

import torch

from app.core import backends
from app.core.config import workspace
from app.core.model_manager import read_meta
from utils.exceptions import IncompatibleModelError, ValidationError
from utils.logger import get_logger
from utils.validators import safe_name

log = get_logger("merging")


def _stage_of_key(key: str, backend=None) -> str:
    """Stage of one checkpoint key, per the owning backend's naming rules."""
    if backend is None:
        from app.core.backends.xurdif import XurdifBackend

        backend = XurdifBackend()
    return backend.stage_of_key(key)


def _blockwise_alpha(key: str, block_weights: dict, alpha: float, backend=None) -> float:
    """Per-key mix weight for a blockwise merge.

    Keys belonging to no stage -- the time embedding MLP, and the scheduler
    buffers that live outside ``denoise_fn.`` -- follow the mean of the block
    sliders rather than ``alpha``. The alpha slider is hidden in blockwise mode,
    so its stale value must never leak in: with all blocks at 0 every tensor has
    to come out as model A exactly, and at 1 as model B.
    """
    stage = _stage_of_key(key, backend)
    if stage in block_weights:
        return float(block_weights[stage])
    vals = [float(v) for v in block_weights.values()]
    return sum(vals) / len(vals) if vals else alpha


def _slerp(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    a_f, b_f = a.flatten().float(), b.flatten().float()
    na, nb = a_f.norm(), b_f.norm()
    if na < 1e-8 or nb < 1e-8:
        return torch.lerp(a.float(), b.float(), alpha).to(a.dtype)
    dot = (a_f / na * (b_f / nb)).sum().clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so < 1e-7:
        return torch.lerp(a.float(), b.float(), alpha).to(a.dtype)
    res = (torch.sin((1 - alpha) * omega) / so) * a_f + (torch.sin(alpha * omega) / so) * b_f
    return res.reshape(a.shape).to(a.dtype)


def check_compat(path_a: str, path_b: str) -> dict:
    ma, mb = read_meta(path_a), read_meta(path_b)
    reasons = []
    # Weights can only be blended between models of literally the same shape, so
    # two engines' checkpoints are never mergeable -- the formats and the UNet
    # topologies are unrelated. Say so plainly rather than failing on a key diff.
    if ma.backend != mb.backend:
        reasons.append(f"different backends ({ma.backend} vs {mb.backend})")
    elif not backends.get(ma.backend).capabilities.merge:
        reasons.append(f"the {ma.backend} backend does not support merging")
    if ma.mtype != mb.mtype:
        reasons.append(f"architectures differ ({ma.mtype} vs {mb.mtype})")
    if list(ma.mults) != list(mb.mults):
        reasons.append(f"channel multipliers differ ({ma.mults} vs {mb.mults})")
    # For the configurable-attention model the layout decides which tensors
    # exist, so it is part of the shape. Both descriptors carry it canonical.
    if ma.mtype == mb.mtype and ma.attn != mb.attn:
        reasons.append(f"attention layouts differ ({ma.attn or 'unrecorded'} vs "
                       f"{mb.attn or 'unrecorded'})")
    return {
        "compatible": len(reasons) == 0,
        "reasons": reasons,
        "a": ma.to_dict(),
        "b": mb.to_dict(),
    }


def _merge_state(sa: dict, sb: dict, method: str, alpha: float, block_weights: dict,
                 backend=None) -> dict:
    keys_a, keys_b = set(sa), set(sb)
    common = keys_a & keys_b
    if not common:
        raise IncompatibleModelError("checkpoints share no tensors; cannot merge")
    out = {}
    for k in sa:
        if k not in sb:
            out[k] = sa[k]
            continue
        ta, tb = sa[k], sb[k]
        if not torch.is_tensor(ta) or not torch.is_tensor(tb) or ta.shape != tb.shape:
            out[k] = ta
            continue
        if method == "blockwise":
            a = _blockwise_alpha(k, block_weights, alpha, backend)
        else:
            a = alpha
        if method == "slerp":
            out[k] = _slerp(ta, tb, a)
        else:
            out[k] = torch.lerp(ta.float(), tb.float(), a).to(ta.dtype)
    return out


#: "Weights to merge": both slots, or one of xurdif's ema / model pair.
WHICH = ("both", "ema", "model")


def blended_slots(slots: dict, which: str) -> set:
    """The slot names a merge blends; every other slot is copied from A.

    ``which`` names xurdif's ema / model pair. A backend whose checkpoint has
    neither (Diffusers' single ``unet`` set) blends everything it has, since
    there is no second set to keep from A.
    """
    if which == "both" or which not in slots:
        return set(slots)
    return {which}


def merge(
    path_a: str,
    path_b: str,
    out_name: str,
    method: str = "linear",
    alpha: float = 0.5,
    block_weights: dict | None = None,
    which: str = "both",
    out_dir: str | None = None,
) -> dict:
    if method not in ("linear", "slerp", "blockwise"):
        raise ValidationError(f"unknown merge method: {method}")
    if which not in WHICH:
        raise ValidationError(f"unknown weights choice: {which}")
    compat = check_compat(path_a, path_b)
    if not compat["compatible"]:
        raise IncompatibleModelError("; ".join(compat["reasons"]))

    out_name = safe_name(out_name, "merged model name")
    block_weights = block_weights or {}
    backend, ref_a = backends.resolve(path_a)
    _, ref_b = backends.resolve(path_b)

    # Reading and writing the checkpoint is the backend's business; the blend
    # itself is plain arithmetic on matching tensors and is shared by both.
    slots_a = backend.merge_slots(ref_a)
    slots_b = backend.merge_slots(ref_b)

    merge_info = {"method": method, "alpha": alpha,
                  "block_weights": block_weights, "which": which}
    merged_from = [_display_name(path_a), _display_name(path_b)]

    blend = blended_slots(slots_a, which)
    slots = {}
    for slot, sa in slots_a.items():
        sb = slots_b.get(slot)
        slots[slot] = (_merge_state(sa, sb, method, alpha, block_weights, backend)
                       if sb is not None and slot in blend else sa)
    if not slots:
        raise IncompatibleModelError("these models expose no weights to merge")

    dest_dir = Path(out_dir) if out_dir else workspace.models
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = backend.write_merged(
        ref_a, slots, dest_dir, out_name,
        {"merged_from": merged_from, "merge": merge_info},
    )
    log.info("merged %s + %s -> %s", merged_from[0], merged_from[1], Path(dest).name)
    return {"path": dest, "name": out_name, **merge_info, "merged_from": merged_from,
            "slots": sorted(slots), "blended": sorted(blend & set(slots_b))}


def _display_name(locator: str) -> str:
    """Short label for a model in a merge record.

    A xurdif model is a file, a Diffusers one may be a repo id -- both read
    better as their last segment than as a full path.
    """
    p = Path(locator)
    return p.name if p.exists() else str(locator).rstrip("/").split("/")[-1]
