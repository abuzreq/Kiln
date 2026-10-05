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
