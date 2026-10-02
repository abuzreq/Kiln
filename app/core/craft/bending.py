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
    # "skip:2", "mid_attn.qkv:v": only looked up when a target asks for one.
    slices = backend.slice_points(model) if any(":" in str(t) for t in targets or []) else {}
    out: set[str] = set()
    for t in targets or []:
        if t in modules or t in slices:
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
        backend = self.backend or _default_backend()
        slices = None
        by_module: dict[str, list[dict]] = defaultdict(list)
        for b in self.bends:
            for name in _resolve_targets(b.get("targets", []), model, backend):
                by_module[name].append(b)
        for name, bends in by_module.items():
            if name in modules:
                h = modules[name].register_forward_hook(self._make_hook(bends))
            else:
                # A slice target: part of some module's input or output.
                slices = slices if slices is not None else backend.slice_points(model)
                spec = slices[name]
                mod = modules[spec["module"]]
                if spec["hook"] == "pre":
                    h = mod.register_forward_pre_hook(self._make_pre_slice_hook(bends, spec))
                else:
                    h = mod.register_forward_hook(self._make_slice_hook(bends, spec))
            self._handles.append(h)
        return self

    def _mine(self) -> bool:
        caller = getattr(_caller, "run", _NOBODY)
        return caller is _NOBODY or caller is self      # else another run's forward

    def _apply(self, bends, y):
        frac = self._frac()
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

    def _make_hook(self, bends):
        def hook(m, inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() != 4 or not self._mine():
                return out
            return self._apply(bends, out)

        return hook

    @staticmethod
    def _bend_slice(x, spec, fn):
        """``x`` with channels [start, stop) replaced by ``fn`` of themselves."""
        start, stop = spec["start"], spec.get("stop") or x.shape[1]
        part = fn(x[:, start:stop])
        if part.shape != x[:, start:stop].shape:
            return x                    # an op that changed the shape cannot go back in
        # Written into a clone rather than re-concatenated: the clone keeps x's
        # memory layout. A Diffusers up block can hand its resnet a
        # channels-last tensor, and a contiguous one in its place sends the
        # conv down another kernel, so even multiply 1 drifted the output.
        y = x.clone()
        y[:, start:stop] = part
        return y

    def _make_slice_hook(self, bends, spec):
        def hook(m, inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() != 4 or not self._mine():
                return out
            return self._bend_slice(out, spec, lambda y: self._apply(bends, y))

        return hook

    def _make_pre_slice_hook(self, bends, spec):
        # The decoder block is called block(x, t): bend the skip half of x and
        # hand t through untouched.
        def hook(m, args):
            if not args or not isinstance(args[0], torch.Tensor) or args[0].dim() != 4 \
                    or not self._mine():
                return None
            x = self._bend_slice(args[0], spec, lambda y: self._apply(bends, y))
            return (x, *args[1:])

        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def build_runtime(bends, meta=None, backend=None) -> BendRuntime:
    return BendRuntime(bends, backend=backend)
