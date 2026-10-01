"""Hook-based model bending runtime.

A *bend stack* is an ordered list of bends. Each bend targets one or more layers
(by id, or by group: all / encoder / mid / decoder / attention / blocks), applies
one op with params, is scheduled over a fraction of the sampling run
(step_start..step_end), and can be toggled ``active``.

At sample time a ``BendRuntime`` registers forward hooks on the targeted modules
and rewrites their output activations. To see what a bend does, use Compare
samples or the bend parameter sweep — both render real generations.
"""
import threading
from collections import defaultdict
from contextlib import contextmanager

import torch

from app.core.craft.ops import apply_op

GROUPS = ("all", "encoder", "mid", "decoder", "attention", "blocks")

# Which sampling run the model forward on this thread belongs to. Hooks live on
# the model's modules, and every run on a model shares one cached module, so a
# bent run's hooks would otherwise fire on another run's forward passes.
_caller = threading.local()
_NOBODY = object()


@contextmanager
def forward_of(runtime):
    """Run the enclosed model call as ``runtime``'s (None: a run with no bends).

    The sampler wraps each denoise step in this. A hook fires only when its own
    runtime is the caller -- or when no sampler run is calling at all, which is
    how ``introspect.capture_activation`` and other direct callers still work.
    """
    prev = getattr(_caller, "run", _NOBODY)
    _caller.run = runtime
    try:
        yield
    finally:
        _caller.run = prev


def _default_backend():
    from app.core.backends.xurdif import XurdifBackend

    return XurdifBackend()


def _resolve_targets(targets, model, backend=None) -> set[str]:
    """Expand a bend's targets to concrete module names.

    The group vocabulary ("encoder", "attention", ...) is shared policy; the
    module names and class names behind it are the backend's, which is why the
    matchers come from there rather than being spelled out here.
    """
    backend = backend or _default_backend()
    modules = dict(model.named_modules())
    points = backend.bend_points(model)
    out: set[str] = set()
    for t in targets or []:
        if t in modules:
            out.add(t)
        elif t == "all":
            out.update(points)
        elif t in ("encoder", "mid", "decoder"):
            out.update(n for n in points if backend.stage_of_point(n) == t)
        elif t == "attention":
            out.update(n for n in points
                       if type(modules.get(n)).__name__ in backend.attention_types)
        elif t == "blocks":
            out.update(n for n in points
                       if type(modules.get(n)).__name__ in backend.block_types)
    return out


class BendRuntime:
    def __init__(self, bends: list[dict], backend=None):
        self.bends = [b for b in (bends or []) if b.get("active", True)]
        self.backend = backend
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
            for name in _resolve_targets(b.get("targets", []), model, self.backend):
                by_module[name].append(b)
        for name, bends in by_module.items():
            h = modules[name].register_forward_hook(self._make_hook(bends))
            self._handles.append(h)
        return self

    def _make_hook(self, bends):
        def hook(m, inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() != 4:
                return out
            caller = getattr(_caller, "run", _NOBODY)
            if caller is not _NOBODY and caller is not self:
                return out                  # another run's forward pass
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


def build_runtime(bends, meta=None, backend=None) -> BendRuntime:
    return BendRuntime(bends, backend=backend)
