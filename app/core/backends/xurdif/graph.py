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
    # ``down_attns`` does not start with ``downs``: match it on its own or the
    # conf model's encoder attention silently lands in "other".
    if name.startswith(("downs", "down_attns")) or name == "init_conv":
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
    if name.startswith(("init_conv", "downs", "down_attns")):
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
        down_attns = getattr(model, "down_attns", None)
        for i in range(len(model.downs)):
            order.append(f"downs.{i}.0")
            # The conf model runs attention after the level's block and before
            # its downsample. An Identity there is a level without attention:
            # nothing to bend, so it is not a point.
            if down_attns is not None and i < len(down_attns) \
                    and not isinstance(down_attns[i], nn.Identity):
                order.append(f"down_attns.{i}")
            order.append(f"downs.{i}.1")
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
    "FullAttention2d": "attention",
    "LinearAttention2d": "attention",
    "WindowAttention2d": "attention",
}

# Which flavour of attention a class is, for the tooltip. The old model's
# single kind is "full" in everything but name.
_ATTN_KIND = {
    "SelfAttention2d": "full",
    "FullAttention2d": "full",
    "LinearAttention2d": "linear",
    "WindowAttention2d": "window",
}

_FIXED_LABEL = {
    "init_conv": "input conv",
    "final_conv": "output conv",
    "mid_block1": "mid block 1",
    "mid_attn": "mid attention",
    "mid_block2": "mid block 2",
}

# Module class names behind the group chips Craft offers.
ATTENTION_TYPES = tuple(_ATTN_KIND)
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
    if len(parts) == 2 and parts[0] == "down_attns":
        return f"enc {parts[1]} · attention"
    if len(parts) == 3 and parts[0] in ("downs", "ups"):
        stage = "enc" if parts[0] == "downs" else "dec"
        # downs.i = [ConvBlock, downsample conv]; ups.i = [upsample conv, ConvBlock]
        role = ({"0": "block", "1": "down"} if parts[0] == "downs"
                else {"0": "up", "1": "block"}).get(parts[2], parts[2])
        return f"{stage} {parts[1]} · {role}"
    return name.replace("downs.", "enc").replace("ups.", "dec").replace(".", " ")


# What bending each point inside a level touches, for the map's hover card.
# ``.conv`` carries a warning: the LayerNorm right after it normalises across
# channels at every pixel, so a uniform Multiply or Add there is undone.
_INNER_ROLES = {
    "conv": "the convolution, before normalisation (a uniform Multiply or Add here is undone by the norm)",
    "norm": "normalised, before the time step's scale and shift",
    "film": "after the time step's scale and shift, before the activation",
    "q": "attention queries: where each position looks",
    "k": "attention keys: what each position is found by",
    "v": "attention values: what gets carried across",
    "out": "the attention's output projection",
}
_QKV_NAMES = ("qkv", "to_qkv")
_OUT_NAMES = ("proj", "to_out")


def _qkv_parts(name: str, mod: nn.Module) -> list[dict]:
    """q, k and v of a conf attention's fused projection, as channel thirds.

    Every conf attention class does ``qkv(x).chunk(3, dim=1)``, so the three
    are contiguous channel ranges of one output: bending one is an output hook
    on that conv that touches its third and leaves the other two alone.
    """
    c = getattr(mod, "out_channels", 0) // 3
    if not c:
        return []
    return [{"id": f"{name}:{part}", "module": name, "hook": "out",
             "start": k * c, "stop": (k + 1) * c, "role": part}
            for k, part in enumerate("qkv")]


def skip_points(model: nn.Module) -> list[dict]:
    """Each skip connection, as a bend target of its own.

    The tensor an encoder level pushes onto ``skips`` also carries on down the
    main path, so bending its source module bends both. Bending only the skip
    means rewriting the skip half of the decoder block's input after
    ``cat((up(x), skip))``: a pre-hook on that block, on channels from
    ``up.out_channels`` on. ``from`` and ``to`` are the main points the map
    draws the arc between.
    """
    downs, ups = getattr(model, "downs", None), getattr(model, "ups", None)
    if downs is None or ups is None or len(downs) != len(ups):
        return []
    down_attns = getattr(model, "down_attns", None)
    out = []
    for i in range(len(downs)):
        j = len(downs) - 1 - i
        split = getattr(ups[j][0], "out_channels", None)
        if split is None:
            continue
        has_attn = down_attns is not None and i < len(down_attns) \
            and not isinstance(down_attns[i], nn.Identity)
        out.append({
            "id": f"skip:{i}", "module": f"ups.{j}.1", "hook": "pre", "start": split, "stop": None,
            "from": f"down_attns.{i}" if has_attn else f"downs.{i}.0", "to": f"ups.{j}.1",
            "level": i,
        })
    return out


def slice_points(model: nn.Module) -> dict[str, dict]:
    """Targets that bend part of a module's input or output rather than a
    whole module output: skips and conf's q/k/v. Keyed by target id."""
    out = {s["id"]: s for s in skip_points(model)}
    for name, mod in model.named_modules():
        if name.rsplit(".", 1)[-1] in _QKV_NAMES and isinstance(mod, nn.Conv2d):
            out.update({p["id"]: p for p in _qkv_parts(name, mod)})
    return out


def inner_points(model: nn.Module, points: list[str]) -> list[dict]:
    """The layers inside each block and attention, in forward order.

    ``.act`` is left out: it is the block's own output, already a point.
    """
    modules = dict(model.named_modules())
    out = []
    for parent in points:
        mod = modules[parent]
        kind = type(mod).__name__
        if kind in BLOCK_TYPES:
            for role in ("conv", "norm", "film"):
                if f"{parent}.{role}" in modules:
                    out.append({"id": f"{parent}.{role}", "module": f"{parent}.{role}",
                                "parent": parent, "role": role})
        elif kind in ATTENTION_TYPES:
            for child in ("q", "k", "v"):
                if isinstance(modules.get(f"{parent}.{child}"), nn.Conv2d):
                    out.append({"id": f"{parent}.{child}", "module": f"{parent}.{child}",
                                "parent": parent, "role": child})
            for child in _QKV_NAMES:
                fused = modules.get(f"{parent}.{child}")
                if isinstance(fused, nn.Conv2d):
                    out += [{**p, "parent": parent} for p in _qkv_parts(f"{parent}.{child}", fused)]
            for child in _OUT_NAMES:
                if isinstance(modules.get(f"{parent}.{child}"), nn.Conv2d):
                    out.append({"id": f"{parent}.{child}", "module": f"{parent}.{child}",
                                "parent": parent, "role": "out"})
    return out


def layer_graph(model: nn.Module, image_size: int = 64) -> dict:
    modules = dict(model.named_modules())
    points = _ordered_points(model)
    inner = inner_points(model, points)
    skips = skip_points(model)

    shapes: dict[str, tuple] = {}
    handles = []
    for name in dict.fromkeys(points + [p["module"] for p in inner]):
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
            "attn_kind": _ATTN_KIND.get(tname),
        })
    by_id = {n["id"]: n for n in nodes}

    def where(shape):
        h = shape[2] if shape and len(shape) >= 3 else None
        return {"h": h, "w": shape[3] if shape and len(shape) >= 4 else None,
                "down_factor": round(image_size / h, 2) if h else None}

    inner_nodes = []
    for p in inner:
        shp = shapes.get(p["module"])
        parent = by_id[p["parent"]]
        c = shp[1] if shp and len(shp) >= 2 else None
        if p.get("hook") == "out" and c:      # one third of a fused qkv
            c = p["stop"] - p["start"]
        inner_nodes.append({
            "id": p["id"],
            "label": f"{parent['label']} · {p['role']}",
            "type": p["role"],
            "role": p["role"],
            "about": _INNER_ROLES.get(p["role"], ""),
            "parent": p["parent"],
            "stage": parent["stage"],
            "channels": c,
            **where(shp),
        })

    skip_nodes = []
    for s in skips:
        src = by_id.get(s["from"])
        dst = by_id.get(s["to"])
        if not src or not dst:
            continue
        # Named by resolution, not by block index: "dec 2" in a layer label
        # counts decoder blocks from the bottom, while the map names levels by
        # resolution, so "enc 1 → dec 2" read as a skip to the wrong level.
        df = src["down_factor"]
        res = "full res" if not df or df <= 1 else f"1/{round(df)} res"
        skip_nodes.append({
            "id": s["id"],
            "label": f"skip · {res}",
            "type": "skip",
            "stage": "skip",
            "from": s["from"],
            "to": s["to"],
            "channels": src["channels"],
            "h": src["h"],
            "w": src["w"],
            "down_factor": src["down_factor"],
        })

    return {"nodes": nodes, "order": points, "ref_size": image_size,
            "inner": inner_nodes, "skips": skip_nodes}
