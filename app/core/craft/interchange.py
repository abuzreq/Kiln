"""Import and export bend stacks in the common network-bending JSON format.

The shape is fixed by the tools this needs to interoperate with:

    {
      "bends": [
        {"path": "input_blocks.4.0.in_layers.0",
         "module_type": "rotate",
         "module_args": {"angle_degrees": 90}}
      ],
      "steps_min": null,
      "steps_max": null,
      "max_denoising_steps": 200,
      "selected_part": "diffusion_model"
    }

Three mismatches with Kiln's own format are worth knowing about, because each one
loses or invents information:

- **One path per entry.** A Kiln bend targets a *set* of layers, so export fans a
  bend out into one entry per layer, and import folds identical consecutive
  entries back into a single bend.
- **The schedule is global.** Kiln schedules each bend separately; this format has
  one ``steps_min``/``steps_max`` for the whole file. Export therefore emits the
  union of the active bends' windows and reports the loss when they disagree.
- **Layer paths are architecture-specific.** The example above is Stable
  Diffusion UNet naming; Kiln models use their own (``downs.1.0``, ``mid_attn``).
  Paths are written and read verbatim -- a file from another tool will parse, but
  its paths will not match, so import reports how many actually resolved.
"""
from app.core.craft import bending, ops

# Kiln's parameter names, where the interchange format spells them differently.
# Anything not listed passes through unchanged.
ARG_ALIASES = {
    "rotate": {"angle": "angle_degrees"},
}

SELECTED_PART = "diffusion_model"      # Kiln only ever bends the UNet
DEFAULT_MAX_STEPS = 200


def _out_args(op: str, params: dict) -> dict:
    alias = ARG_ALIASES.get(op, {})
    return {alias.get(k, k): v for k, v in (params or {}).items()}


def _in_args(op: str, args: dict) -> dict:
    rev = {v: k for k, v in ARG_ALIASES.get(op, {}).items()}
    return {rev.get(k, k): v for k, v in (args or {}).items()}


def _window(bends: list[dict]) -> tuple[float, float, bool]:
    """Union of the bends' schedule windows, plus whether they disagreed."""
    starts = [float(b.get("step_start", 0.0)) for b in bends]
    ends = [float(b.get("step_end", 1.0)) for b in bends]
    if not starts:
        return 0.0, 1.0, False
    differ = len(set(starts)) > 1 or len(set(ends)) > 1
    return min(starts), max(ends), differ


def to_external(bends: list[dict], max_denoising_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Convert a Kiln bend stack to the interchange format.

    ``targets`` are expected to be concrete layer names. Group names are written
    through unchanged and flagged, since no other tool knows what "encoder" means.
    Kiln's slice targets ("skip:2", "mid_attn.qkv:v") are not layer paths at all
    -- they bend part of a layer's input or output -- so they are left out and
    counted instead of written as paths that would match nothing elsewhere.
    """
    steps = max(1, int(max_denoising_steps or DEFAULT_MAX_STEPS))
    active = [b for b in (bends or []) if b.get("active", True)]
    skipped = len(bends or []) - len(active)

    entries = []
    groups_written = []
    kiln_only = []
    for b in active:
        op = b.get("op")
        args = _out_args(op, b.get("params"))
        for target in b.get("targets") or []:
            if ":" in str(target):
                kiln_only.append(target)
                continue
            if target in bending.GROUPS:
                groups_written.append(target)
            entries.append({"path": target, "module_type": op, "module_args": args})

    start, end, differ = _window(active)
    doc = {
        "bends": entries,
        "steps_min": None if start <= 0 else round(start * steps),
        "steps_max": None if end >= 1 else round(end * steps),
        "max_denoising_steps": steps,
        "selected_part": SELECTED_PART,
    }
    report = {
        "bends_written": len(entries),
        "from_stack": len(active),
        "skipped_inactive": skipped,
        "schedule_flattened": differ,
        "unexpanded_groups": sorted(set(groups_written)),
        "kiln_only_targets": sorted(set(kiln_only)),
    }
    return doc, report


def from_external(doc: dict, known_layers: list[str] | None = None) -> tuple[list[dict], dict]:
    """Convert an interchange document into a Kiln bend stack.

    Consecutive entries sharing an operation and arguments are folded back into
    one bend with several targets, which is how they were almost certainly
    written out in the first place.
    """
    if not isinstance(doc, dict):
        raise ValueError("expected a JSON object")
    raw = doc.get("bends")
    if not isinstance(raw, list):
        raise ValueError("missing a 'bends' array")

    catalog = {o["name"] for o in ops.catalog()}
    steps = max(1, int(doc.get("max_denoising_steps") or DEFAULT_MAX_STEPS))
    smin, smax = doc.get("steps_min"), doc.get("steps_max")
    step_start = 0.0 if smin is None else max(0.0, min(1.0, float(smin) / steps))
    step_end = 1.0 if smax is None else max(0.0, min(1.0, float(smax) / steps))
    if step_end < step_start:
        step_start, step_end = step_end, step_start

    known = set(known_layers or [])
    out: list[dict] = []
    unknown_ops: list[str] = []
    matched = unmatched = 0

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        op = entry.get("module_type")
        path = entry.get("path")
        if op not in catalog:
            if op:
                unknown_ops.append(op)
            continue
        if not path:
            continue
        if known:
            if path in known or path in bending.GROUPS:
                matched += 1
            else:
                unmatched += 1
        params = _in_args(op, entry.get("module_args"))
        prev = out[-1] if out else None
        # fold the fan-out back up
        if prev and prev["op"] == op and prev["params"] == params:
            if path not in prev["targets"]:
                prev["targets"].append(path)
            continue
        out.append({
            "op": op,
            "params": params,
            "targets": [path],
            "step_start": step_start,
            "step_end": step_end,
            "active": True,
        })

    report = {
        "bends": len(out),
        "entries_read": len(raw),
        "unknown_ops": sorted(set(unknown_ops)),
        "layers_matched": matched,
        "layers_unmatched": unmatched,
        "checked_layers": bool(known),
    }
    return out, report
