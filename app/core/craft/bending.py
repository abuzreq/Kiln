"""Hook-based model bending runtime.

A *bend stack* is an ordered list of bends. Each bend targets one or more layers
(by id, or by group: all / encoder / mid / decoder / attention / blocks), applies
one op with params, is scheduled over a fraction of the sampling run
(step_start..step_end), and can be toggled ``active``.

At sample time a ``BendRuntime`` registers forward hooks on the targeted modules
and rewrites their output activations. To see what a bend does, use Compare
samples or the bend parameter sweep — both render real generations.
"""
from collections import defaultdict

import torch

from app.core.craft.introspect import _ordered_points, _stage_of
from app.core.craft.ops import apply_op

GROUPS = ("all", "encoder", "mid", "decoder", "attention", "blocks")


def _resolve_targets(targets, model) -> set[str]:
    modules = dict(model.named_modules())
    points = _ordered_points(model)
    out: set[str] = set()
    for t in targets or []:
        if t in modules:
            out.add(t)
        elif t == "all":
            out.update(points)
        elif t in ("encoder", "mid", "decoder"):
            out.update(n for n in points if _stage_of(n) == t)
        elif t == "attention":
            out.update(n for n in points if type(modules.get(n)).__name__ == "SelfAttention2d")
        elif t == "blocks":
            out.update(n for n in points if type(modules.get(n)).__name__ == "ConvBlock")
    return out


class BendRuntime:
    def __init__(self, bends: list[dict]):
        self.bends = [b for b in (bends or []) if b.get("active", True)]
        self.total = 1
        self.cur = 0
        self._handles = []

    def set_total(self, n: int):
        self.total = max(int(n), 1)

    def set_step(self, i: int):
        self.cur = int(i)

    def _frac(self) -> float:
        return self.cur / max(self.total - 1, 1)

    def attach(self, model):
        modules = dict(model.named_modules())
        by_module: dict[str, list[dict]] = defaultdict(list)
        for b in self.bends:
            for name in _resolve_targets(b.get("targets", []), model):
                by_module[name].append(b)
        for name, bends in by_module.items():
            h = modules[name].register_forward_hook(self._make_hook(bends))
            self._handles.append(h)
        return self

    def _make_hook(self, bends):
        def hook(m, inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() != 4:
                return out
            frac = self._frac()
            y = out
            for b in bends:
                s = float(b.get("step_start", 0.0))
                e = float(b.get("step_end", 1.0))
                if frac < s or frac > e:
                    continue
                try:
                    y = apply_op(b["op"], y, b.get("params", {}), {"step": self.cur, "total": self.total})
                except Exception:
                    continue
            return y

        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def build_runtime(bends, meta=None) -> BendRuntime:
    return BendRuntime(bends)
