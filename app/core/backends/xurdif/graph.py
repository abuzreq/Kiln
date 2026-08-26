"""The xurdif UNet's layer vocabulary.

Moved verbatim from ``craft/introspect.py`` and ``craft/merging.py``. These
names -- ``init_conv``, ``downs.i.0``, ``mid_attn``, ``ups.i.1``, ``final_conv``
-- are specific to the vendored architectures, which is exactly why they belong
to a backend rather than to Craft. A Diffusers UNet names the same structural
positions ``conv_in``, ``down_blocks.i.resnets.j``, ``mid_block``, ``up_blocks``,
``conv_out``, and the old matchers silently return nothing for it.
"""
import torch
import torch.nn as nn

_DENOISE_PREFIX = "denoise_fn."


def _stage_of(name: str) -> str:
    if name.startswith("downs") or name == "init_conv":
        return "encoder"
    if name.startswith("mid"):
        return "mid"
    if name.startswith("ups") or name == "final_conv":
        return "decoder"
    return "other"


def stage_of_key(key: str) -> str:
    """Stage of one checkpoint key.

    Checkpoints hold ``GaussianDiffusion.state_dict()``, so every UNet tensor
    arrives prefixed with ``denoise_fn.``. Match against the bare module name or
    nothing ever matches and every tensor silently falls through to 'other'.
    """
    name = key[len(_DENOISE_PREFIX):] if key.startswith(_DENOISE_PREFIX) else key
    if name.startswith("init_conv") or name.startswith("downs"):
        return "encoder"
    if name.startswith("mid"):
        return "mid"
    if name.startswith("ups") or name.startswith("final_conv"):
        return "decoder"
    return "other"


def _ordered_points(model: nn.Module) -> list[str]:
    """Return module names in forward-pass order (the interesting bend points)."""
    order = ["init_conv"]
    if hasattr(model, "downs"):
        for i in range(len(model.downs)):
            order += [f"downs.{i}.0", f"downs.{i}.1"]
    for m in ("mid_block1", "mid_attn", "mid_block2"):
        if hasattr(model, m):
            order.append(m)
    if hasattr(model, "ups"):
        for i in range(len(model.ups)):
            order += [f"ups.{i}.0", f"ups.{i}.1"]
    order.append("final_conv")

    have = dict(model.named_modules())
    return [n for n in order if n in have]


_TYPE_LABEL = {
    "Conv2d": "conv",
    "ConvTranspose2d": "upsample",
    "ConvBlock": "block",
    "SelfAttention2d": "attention",
}

_FIXED_LABEL = {
    "init_conv": "input conv",
    "final_conv": "output conv",
    "mid_block1": "mid block 1",
    "mid_attn": "mid attention",
    "mid_block2": "mid block 2",
}

# Module class names behind the group chips Craft offers.
ATTENTION_TYPES = ("SelfAttention2d",)
BLOCK_TYPES = ("ConvBlock",)


def _label_of(name: str) -> str:
    """Human-readable name for a bend point.

    The dotted module path is the wire id and stays as ``id``; this is what the
    map tooltip and the target chips show, so it has to read like something an
    artist would say out loud ("dec 1 · block"), not like a state-dict key.
    """
    if name in _FIXED_LABEL:
        return _FIXED_LABEL[name]
    parts = name.split(".")
    if len(parts) == 3 and parts[0] in ("downs", "ups"):
        stage = "enc" if parts[0] == "downs" else "dec"
        # downs.i = [ConvBlock, downsample conv]; ups.i = [upsample conv, ConvBlock]
        role = ({"0": "block", "1": "down"} if parts[0] == "downs"
                else {"0": "up", "1": "block"}).get(parts[2], parts[2])
        return f"{stage} {parts[1]} · {role}"
    return name.replace("downs.", "enc").replace("ups.", "dec").replace(".", " ")


def layer_graph(model: nn.Module, image_size: int = 64) -> dict:
    modules = dict(model.named_modules())
    points = _ordered_points(model)

    shapes: dict[str, tuple] = {}
    handles = []
    for name in points:
        mod = modules[name]

        def _hook(m, inp, out, _n=name):
            if isinstance(out, torch.Tensor):
                shapes[_n] = tuple(out.shape)

        handles.append(mod.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = torch.randn(1, 3, image_size, image_size)
            t = torch.tensor([10.0])
            model(x, t)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    nodes = []
    for name in points:
        mod = modules[name]
        tname = type(mod).__name__
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
            "down_factor": round(image_size / h, 2) if h else None,
            "bendable": True,
        })

    return {"nodes": nodes, "order": points, "ref_size": image_size}
