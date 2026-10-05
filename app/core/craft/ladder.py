"""The merge ladder: many blends of two models, sampled on one seed.

A rung is a blend that exists only in memory. ``LadderRig`` reads both models'
weights once and builds one private net; each rung blends the slot the sampler
reads and copies it into that net. N rungs cost N blends and N samples, not N
checkpoints written, loaded and built.

The blend is ``merging._merge_state``, the same arithmetic ``merge()`` saves, so
a rung samples exactly like the model it would become.
"""
from app.core import backends
from app.core.craft.merging import _merge_state
from utils.exceptions import ValidationError

METHODS = ("linear", "slerp", "blockwise")
STAGES = ("encoder", "mid", "decoder")
AXIS_PARAMS = ("alpha", "method", *STAGES)
#: A 5 x 5 block-wise grid, or a 9-step ladder compared on two methods.
MAX_CELLS = 25


def recipe(method: str, alpha: float = 0.5, block_weights: dict | None = None) -> dict:
    """One blend, written the way a cell, a card and a saved recipe record it.

    Block-wise names all three stages (an unnamed stage would otherwise follow
    the mean of the others) and carries no alpha, which it ignores; linear and
    slerp carry only alpha.
    """
    if method not in METHODS:
        raise ValidationError(f"unknown merge method: {method}")
    if method == "blockwise":
        bw = block_weights or {}
        return {"method": method,
                "block_weights": {s: _unit(bw.get(s, 0.5), s) for s in STAGES}}
    return {"method": method, "alpha": _unit(alpha, "alpha")}


def end_of(r: dict) -> str | None:
    """"a" or "b" when a recipe is exactly that model, else None.

    Exact, not approximate: lerp and slerp at 0 and 1, and block-wise with every
    stage at 0 or 1, reproduce A or B tensor for tensor (smoke_merge checks it),
    so the ladder reuses A's and B's own samples for those cells.
    """
    vals = list(r["block_weights"].values()) if r["method"] == "blockwise" else [r["alpha"]]
    if all(v == 0 for v in vals):
        return "a"
    if all(v == 1 for v in vals):
        return "b"
    return None


def plan(method: str, fixed: dict | None, axes: list, known: list | None = None) -> list[dict]:
    """The cells of one ladder, in the order they should be sampled.

    ``axes`` holds up to two ``{"param", "values"}``; with two, the first runs
    down the rows and the second across the columns. A stage axis makes the
    ladder block-wise; ``method`` as an axis compares linear and slerp.

    ``known`` lists recipes the caller already has samples of, on this seed and
    these settings -- a zoom's end steps, a refined grid's centre.

    Returns every cell as ``{x, y, recipe, same_as, order}``: ``same_as`` is
    "a", "b", "known" or the index of an earlier identical cell, for cells that
    need no sample of their own, and ``order`` ranks the rest middle-out, so the
    steps most likely to be picked land first.
    """
    fixed = fixed or {}
    axes = list(axes or [])
    if len(axes) > 2:
        raise ValidationError("a ladder has at most two axes")
    params = [a.get("param") for a in axes]
    for a in axes:
        if a.get("param") not in AXIS_PARAMS:
            raise ValidationError(f"unknown ladder axis: {a.get('param')}")
        if not a.get("values"):
            raise ValidationError(f"the {a['param']} axis has no values")
    if len(set(params)) != len(params):
        raise ValidationError("two ladder axes vary the same thing")
    staged = any(p in STAGES for p in params)
    if staged:
        method = "blockwise"
    if staged and "method" in params:
        raise ValidationError("a block-wise ladder cannot also vary the method")
    if method == "blockwise" and "alpha" in params:
        raise ValidationError("block-wise merges have no single alpha to vary")
    if "method" in params and method == "blockwise":
        raise ValidationError("the method axis compares linear and slerp")
    if method not in METHODS:
        raise ValidationError(f"unknown merge method: {method}")

    rows = axes[0] if len(axes) == 2 else None
    cols = axes[-1] if axes else None
    ys = rows["values"] if rows else [None]
    xs = cols["values"] if cols else [None]
    if len(ys) * len(xs) > MAX_CELLS:
        raise ValidationError(
            f"a ladder of {len(ys) * len(xs)} cells is too large; the most is {MAX_CELLS}")

    have = {_key(recipe(k.get("method", method), k.get("alpha", 0.5), k.get("block_weights")))
            for k in (known or [])}
    cells, seen = [], {}
    for y, vy in enumerate(ys):
        for x, vx in enumerate(xs):
            m = method
            alpha = fixed.get("alpha", 0.5)
            blocks = dict(fixed.get("block_weights") or {})
            for axis, v in ((rows, vy), (cols, vx)):
                if axis is None:
                    continue
                p = axis["param"]
                if p == "method":
                    if v not in ("linear", "slerp"):
                        raise ValidationError("the method axis takes linear and slerp")
                    m = v
                elif p == "alpha":
                    alpha = v
                else:
                    blocks[p] = v
            r = recipe(m, alpha, blocks)
            key = _key(r)
            same = end_of(r) or ("known" if key in have else None)
            if same is None and key in seen:
                same = seen[key]
            seen.setdefault(key, len(cells))
            cells.append({"x": x, "y": y, "recipe": r, "same_as": same, "order": None})

    def from_middle(c):
        d = 0.0
        for axis, i in ((rows, c["y"]), (cols, c["x"])):
            if axis is not None and axis["param"] != "method":
                d += abs(i - (len(axis["values"]) - 1) / 2)
        return d, c["y"], c["x"]

    todo = sorted((c for c in cells if c["same_as"] is None), key=from_middle)
    for rank, c in enumerate(todo):
        c["order"] = rank
    return cells


def _key(r: dict) -> str:
    return repr(sorted((k, sorted(v.items()) if isinstance(v, dict) else v) for k, v in r.items()))


def _unit(v, name: str) -> float:
    try:
        f = round(float(v), 4)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number") from None
    if not 0.0 <= f <= 1.0:
        raise ValidationError(f"{name} must be between 0 and 1")
    return f


class LadderRig:
    """Two models' weights and one net to sample their blends with.

    The net is the rig's own copy, never ``ModelManager``'s cached model A:
    another run may be sampling A while rungs are swapped in. It is built on
    the calling thread's stream, which for a lane job is the stream every rung
    is then copied and sampled on.
    """

    def __init__(self, path_a: str, path_b: str, device: str, ema: bool = True):
        backend, ref_a = backends.resolve(path_a)
        _, ref_b = backends.resolve(path_b)
        slots_a = backend.merge_slots(ref_a)
        slots_b = backend.merge_slots(ref_b)
        self.backend = backend
        self.slot = backend.sampled_slot(slots_a, ema)
        self.a = slots_a[self.slot]
        self.b = slots_b.get(self.slot)  # merge() keeps A's slot when B lacks it
        net, meta = backend.load(ref_a, device=device, ema=ema)
        _own(net)
        self.bundle = {"model": net, "meta": meta, "backend": backend, "ref": ref_a}

    def apply(self, method: str, alpha: float = 0.5, block_weights: dict | None = None) -> dict:
        """Load one rung's blend into the net; returns the bundle to sample."""
        import torch

        if method not in METHODS:
            raise ValidationError(f"unknown merge method: {method}")
        state = self.a if self.b is None else _merge_state(
            self.a, self.b, method, float(alpha), block_weights or {}, self.backend)
        with torch.no_grad():
            self.backend.load_slot(self.bundle["model"], state)
        return self.bundle


def _own(net):
    """Give the net storage of its own before rungs are copied into it.

    A Diffusers net can come back backed by safetensors' memory map, which is
    not ours to write into; a xurdif net already owns its weights, and the copy
    costs a few MB once per ladder.
    """
    inner = getattr(net, "wrapped", net)
    for p in inner.parameters():
        p.data = p.data.clone()
    for b in inner.buffers():
        b.data = b.data.clone()
