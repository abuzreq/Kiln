"""The Diffusers UNet2DModel layer vocabulary.

Same job as ``backends/xurdif/graph.py``, different names. A xurdif net calls
its stages ``downs`` / ``mid_attn`` / ``ups``; this one calls them
``down_blocks`` / ``mid_block`` / ``up_blocks`` and nests two more levels
(``resnets.j``, ``attentions.j``). Nothing else about bending changes: the ops
are pure functions on (B,C,H,W) tensors and the hook mechanism is shared.

``type`` and ``stage`` deliberately reuse the words the existing UI already
groups on -- ``"block"``, ``"attention"``, ``"encoder"`` / ``"mid"`` /
``"decoder"`` -- so the Craft map, the group chips and the saved bend presets
all work against a Diffusers model without a frontend change.
"""
import torch

_TYPE_LABEL = {
    "Conv2d": "conv",
    "ResnetBlock2D": "block",
    "Attention": "attention",
    "Downsample2D": "conv",
    "Upsample2D": "upsample",
}

# Module class names behind the group chips Craft offers.
ATTENTION_TYPES = ("Attention",)
BLOCK_TYPES = ("ResnetBlock2D",)

_FIXED_LABEL = {
    "conv_in": "input conv",
    "conv_out": "output conv",
}


def _stage_of(name: str) -> str:
    if name.startswith("down_blocks") or name == "conv_in":
        return "encoder"
    if name.startswith("mid_block"):
        return "mid"
    if name.startswith("up_blocks") or name.startswith("conv_out"):
        return "decoder"
    return "other"


def stage_of_key(key: str) -> str:
    """Bucket one state-dict key.

    Unlike xurdif there is no prefix to strip -- a Diffusers checkpoint stores
    the UNet's own state dict directly. Keys outside any stage (the time
    embedding, the output norm) fall through to 'other', same as before.
    """
    if key.startswith("conv_in") or key.startswith("down_blocks"):
        return "encoder"
    if key.startswith("mid_block"):
        return "mid"
    if key.startswith("up_blocks") or key.startswith("conv_out"):
        return "decoder"
    return "other"


def _interleave(prefix: str, resnets, attentions, have) -> list[str]:
    """Resnet/attention pairs in the order the block's forward runs them."""
    out = []
    for j in range(len(resnets)):
        cand = f"{prefix}.resnets.{j}"
        if cand in have:
            out.append(cand)
        cand = f"{prefix}.attentions.{j}"
        if cand in have:
            out.append(cand)
    return out


def _ordered_points(model) -> list[str]:
    """Module names in forward-pass order.

    Derived structurally rather than by tracing, so it is deterministic and
    costs nothing: a UNet2DModel runs its blocks in list order, and inside a
    block the resnets and attentions alternate.
    """
    have = dict(model.named_modules())
    order: list[str] = []

    if "conv_in" in have:
        order.append("conv_in")

    downs = getattr(model, "down_blocks", None) or []
    for i, block in enumerate(downs):
        p = f"down_blocks.{i}"
        order += _interleave(p, getattr(block, "resnets", []) or [],
                             getattr(block, "attentions", []) or [], have)
        if f"{p}.downsamplers.0" in have:
            order.append(f"{p}.downsamplers.0")

    mid = getattr(model, "mid_block", None)
    if mid is not None:
        resnets = getattr(mid, "resnets", []) or []
        attns = getattr(mid, "attentions", []) or []
        if "mid_block.resnets.0" in have:
            order.append("mid_block.resnets.0")
        # UNetMidBlock2D runs resnets[0], then (attn, resnet) pairs.
        for k in range(len(resnets) - 1):
            for cand in (f"mid_block.attentions.{k}", f"mid_block.resnets.{k + 1}"):
                if cand in have:
                    order.append(cand)

    ups = getattr(model, "up_blocks", None) or []
    for i, block in enumerate(ups):
        p = f"up_blocks.{i}"
        order += _interleave(p, getattr(block, "resnets", []) or [],
                             getattr(block, "attentions", []) or [], have)
        if f"{p}.upsamplers.0" in have:
            order.append(f"{p}.upsamplers.0")

    if "conv_out" in have:
        order.append("conv_out")

    return [n for n in order if n in have]


def _label_of(name: str) -> str:
    """Human-readable name for a bend point.

    Reads the way the xurdif labels do ("enc 1 · block"), because the same
    chips and tooltips show both and a user should not have to learn two
    vocabularies to bend two models.
    """
    if name in _FIXED_LABEL:
        return _FIXED_LABEL[name]
    parts = name.split(".")
    stage = {"down_blocks": "enc", "up_blocks": "dec", "mid_block": "mid"}.get(parts[0])
    if stage is None:
        return name.replace(".", " ")

    if parts[0] == "mid_block":
        kind, idx = (parts[1], parts[2]) if len(parts) >= 3 else (parts[-1], "0")
        if kind == "attentions":
            return "mid attention"
        return f"mid block {int(idx) + 1}"

    if len(parts) < 3:
        return name.replace(".", " ")
    i, kind = parts[1], parts[2]
    if kind in ("downsamplers", "upsamplers"):
        return f"{stage} {i} · {'down' if kind == 'downsamplers' else 'up'}"
    idx = parts[3] if len(parts) > 3 else "0"
    role = {"resnets": "block", "attentions": "attn"}.get(kind, kind)
    return f"{stage} {i} · {role} {idx}"


def layer_graph(model, image_size: int = 64) -> dict:
    """Ordered bend points with their channel counts and spatial sizes.

    ``image_size`` is snapped up to a multiple of the net's downsampling factor
    first: a UNet2DModel with six blocks halves the feature map five times, so
    the Craft default of 64 is fine at four blocks and impossible at seven.
    """
    from .loader import size_multiple

    modules = dict(model.named_modules())
    points = _ordered_points(model)

    channels = list(getattr(getattr(model, "config", None), "block_out_channels", None) or [])
    step = size_multiple(channels) if channels else 16
    probe_size = max(step, (int(image_size) // step) * step)

    shapes: dict[str, tuple] = {}
    handles = []
    for name in points:
        def _hook(m, inp, out, _n=name):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.dim() == 4:
                shapes[_n] = tuple(t.shape)

        handles.append(modules[name].register_forward_hook(_hook))

    was_training = getattr(model, "training", False)
    model.eval()
    try:
        with torch.no_grad():
            x = torch.randn(1, 3, probe_size, probe_size)
            t = torch.tensor([10], dtype=torch.long)
            model(x, t)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    nodes = []
    for name in points:
        tname = type(modules[name]).__name__
        shp = shapes.get(name)
        c = shp[1] if shp and len(shp) >= 2 else None
        h = shp[2] if shp and len(shp) >= 3 else None
        w = shp[3] if shp and len(shp) >= 4 else None
        nodes.append({
            "id": name,
            "label": _label_of(name),
            "type": _TYPE_LABEL.get(tname, tname.lower()),
            "stage": _stage_of(name),
            "channels": c,
            "h": h,
            "w": w,
            "down_factor": round(probe_size / h, 2) if h else None,
            "bendable": True,
        })

    return {"nodes": nodes, "order": points, "ref_size": probe_size}
