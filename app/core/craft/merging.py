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

from app.core.config import workspace
from app.core.model_manager import read_meta
from utils.exceptions import IncompatibleModelError, ValidationError
from utils.logger import get_logger
from utils.validators import safe_name

log = get_logger("merging")


_DENOISE_PREFIX = "denoise_fn."


def _stage_of_key(key: str) -> str:
    """Stage of one checkpoint key.

    Checkpoints hold ``GaussianDiffusion.state_dict()``, so every UNet tensor
    arrives prefixed with ``denoise_fn.`` (model_manager.load strips the same
    prefix). Match against the bare module name or nothing ever matches and
    every tensor silently falls through to the 'other' bucket.
    """
    name = key[len(_DENOISE_PREFIX):] if key.startswith(_DENOISE_PREFIX) else key
    if name.startswith("init_conv") or name.startswith("downs"):
        return "encoder"
    if name.startswith("mid"):
        return "mid"
    if name.startswith("ups") or name.startswith("final_conv"):
        return "decoder"
    return "other"


def _blockwise_alpha(key: str, block_weights: dict, alpha: float) -> float:
    """Per-key mix weight for a blockwise merge.

    Keys belonging to no stage -- the time embedding MLP, and the scheduler
    buffers that live outside ``denoise_fn.`` -- follow the mean of the block
    sliders rather than ``alpha``. The alpha slider is hidden in blockwise mode,
    so its stale value must never leak in: with all blocks at 0 every tensor has
    to come out as model A exactly, and at 1 as model B.
    """
    stage = _stage_of_key(key)
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
    if ma.mtype != mb.mtype:
        reasons.append(f"architectures differ ({ma.mtype} vs {mb.mtype})")
    if list(ma.mults) != list(mb.mults):
        reasons.append(f"channel multipliers differ ({ma.mults} vs {mb.mults})")
    return {
        "compatible": len(reasons) == 0,
        "reasons": reasons,
        "a": ma.to_dict(),
        "b": mb.to_dict(),
    }


def _merge_state(sa: dict, sb: dict, method: str, alpha: float, block_weights: dict) -> dict:
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
            a = _blockwise_alpha(k, block_weights, alpha)
        else:
            a = alpha
        if method == "slerp":
            out[k] = _slerp(ta, tb, a)
        else:
            out[k] = torch.lerp(ta.float(), tb.float(), a).to(ta.dtype)
    return out


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
    compat = check_compat(path_a, path_b)
    if not compat["compatible"]:
        raise IncompatibleModelError("; ".join(compat["reasons"]))

    out_name = safe_name(out_name, "merged model name")
    block_weights = block_weights or {}
    da = torch.load(path_a, map_location="cpu", weights_only=False)
    db = torch.load(path_b, map_location="cpu", weights_only=False)

    result = {
        "step": 0,
        "mults": da.get("mults", read_meta(path_a).mults),
        "mtype": da.get("mtype", read_meta(path_a).mtype),
        "pred": da.get("pred", "eps"),
        "merged_from": [Path(path_a).name, Path(path_b).name],
        "merge": {"method": method, "alpha": alpha, "block_weights": block_weights, "which": which},
    }

    for slot in ("model", "ema"):
        if slot in da and slot in db:
            result[slot] = _merge_state(da[slot], db[slot], method, alpha, block_weights)
        elif slot in da:
            result[slot] = da[slot]
    # ensure both slots exist
    if "model" not in result and "ema" in result:
        result["model"] = result["ema"]
    if "ema" not in result and "model" in result:
        result["ema"] = result["model"]

    dest_dir = Path(out_dir) if out_dir else workspace.models
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{out_name}.pt"
    torch.save(result, str(dest))
    log.info("merged %s + %s -> %s", Path(path_a).name, Path(path_b).name, dest.name)
    return {"path": str(dest), "name": out_name, **result["merge"], "merged_from": result["merged_from"]}
