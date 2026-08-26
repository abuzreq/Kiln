"""Introspect a UNet into an ordered, bendable layer graph.

Produces the data the Craft visualizer draws: an ordered pipeline of "bend points"
(modules that emit feature maps), each with type, stage (encoder / mid / decoder),
channel count and spatial size at a reference resolution.

The naming rules themselves belong to whichever backend owns the model -- a
xurdif UNet calls its stages ``downs``/``mid_attn``/``ups`` and a Diffusers one
calls them ``down_blocks``/``mid_block``/``up_blocks`` -- so they live in
``app.core.backends.<name>.graph`` and are re-exported here for callers that
predate the split.
"""
import torch
import torch.nn as nn

from app.core.backends.xurdif.graph import (  # noqa: F401
    _label_of, _ordered_points, _stage_of, _TYPE_LABEL,
)


def introspect(model, image_size: int = 64) -> dict:
    """Layer graph for a model. Prefer ``backend.layer_graph()`` when you have one."""
    from app.core.backends.xurdif import XurdifBackend

    return XurdifBackend().layer_graph(model, image_size=image_size)


def capture_activation(model: nn.Module, node_id: str, x, t, bend_runtime=None):
    """Run one forward and return the output tensor of ``node_id`` (post-bend if given).

    Currently unused: the layer activation preview that called this was removed
    (it was misleading on three counts, none of them here — the hook ordering
    below is correct). Kept as the generic "run a pass, grab one layer" probe.
    """
    modules = dict(model.named_modules())
    if node_id not in modules:
        raise ValueError(f"unknown layer: {node_id}")
    captured = {}

    def _hook(m, inp, out, _n=node_id):
        captured["t"] = out.detach()

    # Attach bend hooks FIRST so our capture hook (registered after) sees the
    # post-bend activation for the target layer.
    #
    # The caller owns the runtime's schedule position: this used to force
    # set_step(0), which pinned every probe to frac 0 and silently skipped any
    # bend whose window starts later (spatial 0.15, morph 0.20, edges 0.45).
    # Set set_total()/set_step() on the runtime before calling in.
    if bend_runtime is not None:
        bend_runtime.attach(model)
    handle = modules[node_id].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(x, t)
    finally:
        handle.remove()
        if bend_runtime is not None:
            bend_runtime.detach()
    return captured.get("t")
