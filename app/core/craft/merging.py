"""Two-way model merging for xurdif checkpoints.

Blends of A and B:
- linear:    out = (1-a)*A + a*B, per tensor
- slerp:     spherical interpolation per tensor (preserves norm/character better)
- blockwise: linear, but with a different alpha per stage (encoder / mid / decoder)

From a base both were trained from, on task vectors tau = model - base. These
are meant for A and B that share that ancestor (see relatedness.py), but any
compatible base is allowed, and with no base at all tau is the weights
themselves -- a zero base -- for experiments:
- task_arithmetic: out = base + strength * ((1-a)*tau_A + a*tau_B)
- ties:      TIES-Merging (Yadav et al. 2023): trim each tau to its top
             ``density`` share by magnitude, elect a sign per entry from the
             weighted sum, and average only the entries that agree with it
- dare_ties: the same, with DARE's random drop and 1/density rescale
             (Yu et al. 2024) in place of the trim

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


BLEND_METHODS = ("linear", "slerp", "blockwise")
BASE_METHODS = ("task_arithmetic", "ties", "dare_ties")
METHODS = BLEND_METHODS + BASE_METHODS
#: strength (lambda) is allowed past 1: adding both deltas at full weight is
#: the community's "add difference", and the point of the knob.
STRENGTH_MAX = 2.0


def _trim(t, density: float):
    """TIES' trim: keep the top ``density`` share of entries by magnitude."""
    if density >= 1:
        return t
    mag = t.abs().flatten()
    k = max(1, round(mag.numel() * density))
    thr = mag.kthvalue(mag.numel() - k + 1).values
    return torch.where(t.abs() >= thr, t, torch.zeros_like(t))


def _dare(t, density: float, seed_key: str):
    """DARE's drop: keep each entry with probability ``density``, rescaled by 1/density.

    The mask is seeded from the model's role, the tensor and the density --
    not from the balance or the strength -- so a saved merge replays its
    ladder rung exactly and a ladder's mask stays put as the balance moves.
    """
    if density >= 1:
        return t
    import zlib

    g = torch.Generator().manual_seed(zlib.crc32(f"{seed_key}|{density}".encode()))
    keep = torch.rand(t.shape, generator=g, dtype=torch.float64) < density
    return torch.where(keep, t / density, torch.zeros_like(t))


def _disjoint(da, db, wa: float, wb: float):
    """TIES' elect and disjoint merge for two task vectors with weights.

    The elected sign is that of the weighted sum; each entry is the weighted
    mean of the deltas that agree with it, so a conflict goes to the larger
    weighted change and an entry only one model kept stays whole.
    """
    elected = torch.sign(wa * da + wb * db)
    ka = (torch.sign(da) == elected) & (da != 0)
    kb = (torch.sign(db) == elected) & (db != 0)
    num = wa * da * ka + wb * db * kb
    den = wa * ka + wb * kb
    return torch.where(den > 0, num / den.clamp_min(1e-300), torch.zeros_like(num))


def _merge_from_base(sa: dict, sb: dict, sc: dict | None, method: str, alpha: float,
                     density: float = 1.0, strength: float = 1.0) -> dict:
    """A task-vector merge of A and B relative to ``sc``, the base's slot.

    ``sc`` None is a zero base: the task vectors are the weights themselves.
    Computed in float64 and cast back. Tensors the three do not share, or that
    differ in shape, are A's.
    """
    out = {}
    for k, ta in sa.items():
        tb = sb.get(k)
        tc = ta.new_zeros(ta.shape) if sc is None and torch.is_tensor(ta) else (sc or {}).get(k)
        if not all(torch.is_tensor(t) for t in (ta, tb, tc)) or not ta.is_floating_point() \
                or not (ta.shape == tb.shape == tc.shape):
            out[k] = ta
            continue
        if sc is None and torch.equal(ta, tb):
            # With no base, what A and B hold alike (a fixed noise schedule,
            # say) is not a change to trim or scale.
            out[k] = ta
            continue
        c = tc.double()
        da, db = ta.double() - c, tb.double() - c
        if method == "task_arithmetic":
            tau = (1 - alpha) * da + alpha * db
        else:
            if method == "ties":
                da, db = _trim(da, density), _trim(db, density)
            else:
                da, db = _dare(da, density, f"a|{k}"), _dare(db, density, f"b|{k}")
            tau = _disjoint(da, db, 1 - alpha, alpha)
        out[k] = (c + strength * tau).to(ta.dtype)
    return out


def check_base_params(density: float, strength: float):
    if not 0 < density <= 1:
        raise ValidationError("density must be above 0 and at most 1")
    if not 0 <= strength <= STRENGTH_MAX:
        raise ValidationError(f"strength must be between 0 and {STRENGTH_MAX:g}")


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
    base: str | None = None,
    density: float = 1.0,
    strength: float = 1.0,
    align: str = "none",
) -> dict:
    """Blend A and B into a new checkpoint.

    ``base`` (for the methods that work from one; None is a zero base) and
    ``align`` ("weights" or "activations": reorder B's units, and the base's,
    to line up with A's first -- see rebasin.py) are options to experiment
    with: neither is refused for a pair it is unlikely to suit.
    """
    from app.core.craft import rebasin

    if method not in METHODS:
        raise ValidationError(f"unknown merge method: {method}")
    if which not in WHICH:
        raise ValidationError(f"unknown weights choice: {which}")
    if align not in rebasin.ALIGN:
        raise ValidationError(f"unknown alignment: {align}")
    from_base = method in BASE_METHODS
    if not from_base:
        base = None
    if from_base:
        check_base_params(density, strength)
    compat = check_compat(path_a, path_b)
    if not compat["compatible"]:
        raise IncompatibleModelError("; ".join(compat["reasons"]))
    if base:
        check_base_compat(path_a, path_b, base)

    out_name = safe_name(out_name, "merged model name")
    block_weights = block_weights or {}
    backend, ref_a = backends.resolve(path_a)
    _, ref_b = backends.resolve(path_b)

    # Reading and writing the checkpoint is the backend's business; the blend
    # itself is plain arithmetic on matching tensors and is shared by both.
    slots_a = backend.merge_slots(ref_a)
    slots_b = backend.merge_slots(ref_b)
    slots_c = backend.merge_slots(backends.resolve(base)[1]) if base else None
    if align != "none":
        slots_b = rebasin.align_slots(slots_b, *rebasin.pair_perms(path_a, path_b, align,
                                                                   _probe_device()))
        if slots_c is not None:
            slots_c = rebasin.align_slots(slots_c, *rebasin.pair_perms(path_a, base, align,
                                                                       _probe_device()))

    merge_info = {"method": method, "alpha": alpha,
                  "block_weights": block_weights, "which": which, "align": align}
    merged_from = [_display_name(path_a), _display_name(path_b)]
    if from_base:
        merge_info.update(base=_display_name(base) if base else None,
                          density=density, strength=strength)
        if base:
            merged_from.append(_display_name(base))

    blend = blended_slots(slots_a, which)
    slots = {}
    for slot, sa in slots_a.items():
        sb = slots_b.get(slot)
        if sb is None or slot not in blend:
            slots[slot] = sa
        elif from_base:
            sc = base_slot(slots_c, slot) if slots_c is not None else None
            slots[slot] = _merge_from_base(sa, sb, sc, method, alpha, density, strength)
        else:
            slots[slot] = _merge_state(sa, sb, method, alpha, block_weights, backend)
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


def _probe_device() -> str:
    from app.core.engine.sampler import pick_device

    return pick_device("auto")


def check_base_compat(path_a: str, path_b: str, base: str):
    """A base must have A's and B's shape too, or there is no task vector to take."""
    for other in (path_a, path_b):
        c = check_compat(other, base)
        if not c["compatible"]:
            raise IncompatibleModelError("base: " + "; ".join(c["reasons"]))


def base_slot(slots_c: dict, slot: str) -> dict:
    """The base's weights to measure a slot against: the same slot, else any it has."""
    return slots_c.get(slot) or next(iter(slots_c.values()))


def _display_name(locator: str) -> str:
    """Short label for a model in a merge record.

    A xurdif model is a file, a Diffusers one may be a repo id -- both read
    better as their last segment than as a full path.
    """
    p = Path(locator)
    return p.name if p.exists() else str(locator).rstrip("/").split("/")[-1]
