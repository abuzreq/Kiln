"""The merge ladder: many blends of two models, sampled on one seed.

A rung is a blend that exists only in memory. ``LadderRig`` reads both models'
weights once and builds one private net; each rung blends the slot the sampler
reads and copies it into that net. N rungs cost N blends and N samples, not N
checkpoints written, loaded and built.

The blend is ``merging._merge_state`` (or ``_merge_from_base`` for the methods
that work from a base), the same arithmetic ``merge()`` saves, so a rung
samples exactly like the model it would become.
"""
import os
import threading

from app.core import backends
from app.core.craft.merging import (
    BASE_METHODS, BLEND_METHODS, METHODS, STRENGTH_MAX, _merge_from_base, _merge_state,
    base_slot,
)
from utils.exceptions import ValidationError

STAGES = ("encoder", "mid", "decoder")
AXIS_PARAMS = ("alpha", "method", "density", "strength", *STAGES)
#: A 5 x 5 block-wise grid, or a 9-step ladder compared on two methods.
MAX_CELLS = 25


def recipe(method: str, alpha: float = 0.5, block_weights: dict | None = None,
           density: float = 1.0, strength: float = 1.0) -> dict:
    """One blend, written the way a cell, a card and a saved recipe record it.

    Block-wise names all three stages (an unnamed stage would otherwise follow
    the mean of the others) and carries no alpha, which it ignores; linear and
    slerp carry only alpha; the base methods carry alpha, density and strength.
    The base itself belongs to the ladder, like A and B, not to the recipe.
    """
    if method not in METHODS:
        raise ValidationError(f"unknown merge method: {method}")
    if method == "blockwise":
        bw = block_weights or {}
        return {"method": method,
                "block_weights": {s: _unit(bw.get(s, 0.5), s) for s in STAGES}}
    if method in BASE_METHODS:
        d = _unit(density, "density")
        if d == 0:
            raise ValidationError("density must be above 0")
        return {"method": method, "alpha": _unit(alpha, "alpha"), "density": d,
                "strength": _number(strength, "strength", 0.0, STRENGTH_MAX)}
    return {"method": method, "alpha": _unit(alpha, "alpha")}


def end_of(r: dict, aligned: bool = False) -> str | None:
    """"a" or "b" when a recipe is exactly that model, else None.

    Exact, not approximate: lerp and slerp at 0 and 1, and block-wise with every
    stage at 0 or 1, reproduce A or B tensor for tensor (smoke_merge checks it),
    so the ladder reuses A's and B's own samples for those cells. A base method
    never claims an end: base + (A - base) is A only up to rounding, and the
    trim or drop makes it something else entirely, so its ends are sampled.
    Nor does an aligned B: it computes B's function in A's unit order, which
    sums in another order and is not B bit for bit -- sampling that end shows
    the alignment kept B.
    """
    if r["method"] in BASE_METHODS:
        return None
    vals = list(r["block_weights"].values()) if r["method"] == "blockwise" else [r["alpha"]]
    if all(v == 0 for v in vals):
        return "a"
    if all(v == 1 for v in vals) and not aligned:
        return "b"
    return None


def _recipe_of(d: dict, method: str) -> dict:
    return recipe(d.get("method", method), d.get("alpha", 0.5), d.get("block_weights"),
                  d.get("density", 1.0), d.get("strength", 1.0))


def plan(method: str, fixed: dict | None, axes: list, known: list | None = None,
         aligned: bool = False) -> list[dict]:
    """The cells of one ladder, in the order they should be sampled.

    ``axes`` holds up to two ``{"param", "values"}``; with two, the first runs
    down the rows and the second across the columns. A stage axis makes the
    ladder block-wise; ``method`` as an axis compares methods of one family
    (linear and slerp, or the base methods); ``density`` and ``strength``
    belong to the base methods.

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
    if method not in METHODS:
        raise ValidationError(f"unknown merge method: {method}")
    from_base = method in BASE_METHODS
    staged = any(p in STAGES for p in params)
    if staged and from_base:
        raise ValidationError("block-wise weights do not apply to merges from a base")
    if staged:
        method = "blockwise"
    if staged and "method" in params:
        raise ValidationError("a block-wise ladder cannot also vary the method")
    if method == "blockwise" and "alpha" in params:
        raise ValidationError("block-wise merges have no single alpha to vary")
    if "method" in params and method == "blockwise":
        raise ValidationError("the method axis compares linear and slerp")
    if not from_base and ({"density", "strength"} & set(params)):
        raise ValidationError("density and strength belong to merges from a base")
    family = BASE_METHODS if from_base else tuple(m for m in BLEND_METHODS if m != "blockwise")

    rows = axes[0] if len(axes) == 2 else None
    cols = axes[-1] if axes else None
    ys = rows["values"] if rows else [None]
    xs = cols["values"] if cols else [None]
    if len(ys) * len(xs) > MAX_CELLS:
        raise ValidationError(
            f"a ladder of {len(ys) * len(xs)} cells is too large; the most is {MAX_CELLS}")

    have = {_key(_recipe_of(k, method)) for k in (known or [])}
    cells, seen = [], {}
    for y, vy in enumerate(ys):
        for x, vx in enumerate(xs):
            m = method
            alpha = fixed.get("alpha", 0.5)
            density = fixed.get("density", 1.0)
            strength = fixed.get("strength", 1.0)
            blocks = dict(fixed.get("block_weights") or {})
            for axis, v in ((rows, vy), (cols, vx)):
                if axis is None:
                    continue
                p = axis["param"]
                if p == "method":
                    if v not in family:
                        raise ValidationError(
                            "the method axis compares " + " and ".join(family))
                    m = v
                elif p == "alpha":
                    alpha = v
                elif p == "density":
                    density = v
                elif p == "strength":
                    strength = v
                else:
                    blocks[p] = v
            r = recipe(m, alpha, blocks, density, strength)
            key = _key(r)
            same = end_of(r, aligned) or ("known" if key in have else None)
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


def _number(v, name: str, lo: float, hi: float) -> float:
    try:
        f = round(float(v), 4)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number") from None
    if not lo <= f <= hi:
        raise ValidationError(f"{name} must be between {lo:g} and {hi:g}")
    return f


def _unit(v, name: str) -> float:
    return _number(v, name, 0.0, 1.0)


class LadderRig:
    """Two models' weights (and a base's) and one net to sample their blends with.

    ``base`` is needed by the base methods only. ``align`` reorders B's units
    (and the base's) to line up with A's first, with the permutation
    ``merge()`` will use for the same pair.

    The net is the rig's own copy, never ``ModelManager``'s cached model A:
    another run may be sampling A while rungs are swapped in. It is built on
    the calling thread's stream, which for a lane job is the stream every rung
    is then copied and sampled on.
    """

    def __init__(self, path_a: str, path_b: str, device: str, ema: bool = True,
                 base: str | None = None, align: str = "none", cancel=None):
        from app.core.craft import rebasin

        backend, ref_a = backends.resolve(path_a)
        _, ref_b = backends.resolve(path_b)
        slots_a = backend.merge_slots(ref_a)
        slots_b = backend.merge_slots(ref_b)
        slots_c = backend.merge_slots(backends.resolve(base)[1]) if base else None
        if align != "none":
            slots_b = rebasin.align_slots(slots_b, *rebasin.pair_perms(
                path_a, path_b, align, device, ema, cancel))
            if slots_c is not None:
                slots_c = rebasin.align_slots(slots_c, *rebasin.pair_perms(
                    path_a, base, align, device, ema, cancel))
        self.backend = backend
        self.slot = backend.sampled_slot(slots_a, ema)
        self.a = slots_a[self.slot]
        self.b = slots_b.get(self.slot)  # merge() keeps A's slot when B lacks it
        self.c = base_slot(slots_c, self.slot) if slots_c is not None else None
        net, meta = backend.load(ref_a, device=device, ema=ema)
        _own(net)
        self.bundle = {"model": net, "meta": meta, "backend": backend, "ref": ref_a}
        self._loaded = None

    def apply(self, method: str, alpha: float = 0.5, block_weights: dict | None = None,
              density: float = 1.0, strength: float = 1.0) -> dict:
        """Load one rung's blend into the net; returns the bundle to sample."""
        import torch

        if method not in METHODS:
            raise ValidationError(f"unknown merge method: {method}")
        # The same blend twice in a row (Generate again on one recipe) is
        # already in the net.
        key = (method, round(float(alpha), 6), tuple(sorted((block_weights or {}).items())),
               round(float(density), 6), round(float(strength), 6))
        if key == self._loaded:
            return self.bundle
        if self.b is None:
            state = self.a
        elif method in BASE_METHODS:
            if self.c is None:
                raise ValidationError("merges from a base need a base model")
            state = _merge_from_base(self.a, self.b, self.c, method, float(alpha),
                                     float(density), float(strength))
        else:
            state = _merge_state(self.a, self.b, method, float(alpha), block_weights or {},
                                 self.backend)
        with torch.no_grad():
            self.backend.load_slot(self.bundle["model"], state)
        self._loaded = key
        return self.bundle


# One rig kept between runs, for merge recipes used from Create: Generate on a
# recipe again should not read two checkpoints and build a net each time. Lanes
# never run two jobs on the same model at once, and a recipe run is queued on
# both A and B, so no two runs share this rig's net. A ladder builds its own.
_shared_lock = threading.Lock()
_shared: tuple | None = None    # (key, LadderRig)


def _stamp(path: str):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def shared_rig(path_a: str, path_b: str, device: str, ema: bool = True,
               base: str | None = None, align: str = "none", cancel=None) -> LadderRig:
    """The rig for A and B (and a base, and an alignment), reused while no file changes."""
    global _shared
    key = (path_a, _stamp(path_a), path_b, _stamp(path_b),
           base, _stamp(base) if base else None, align, device, bool(ema))
    with _shared_lock:
        if _shared is not None and _shared[0] == key:
            return _shared[1]
        _shared = None              # let the old pair go before loading the new one
        rig = LadderRig(path_a, path_b, device, ema=ema, base=base, align=align, cancel=cancel)
        _settle(device)
        _shared = (key, rig)
        return rig


def drop_shared_rig() -> bool:
    """Forget the kept rig (Free GPU, cache clears). True if there was one."""
    global _shared
    with _shared_lock:
        had, _shared = _shared is not None, None
    return had


def _settle(device: str):
    """Finish building on this lane's stream: a later run may be on another lane."""
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.current_stream().synchronize()


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
