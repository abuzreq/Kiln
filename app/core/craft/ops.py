"""Catalog of model-bending operations.

Each op transforms a feature-map activation tensor ``x`` of shape (B, C, H, W)
and returns a tensor of the same shape. Inspired by ComfyUI-Model-Bending plus
xurdif's edge operators. Ops are pure functions registered with a small parameter
schema so the UI can render controls automatically.
"""
import math

import torch
import torch.nn.functional as F

OP_CATALOG: dict[str, dict] = {}


def op(name, label, category, params, help=None):
    def deco(fn):
        OP_CATALOG[name] = {
            "name": name, "label": label, "category": category,
            "params": params, "help": help or "", "fn": fn,
        }
        return fn
    return deco


def _p(name, label, default, lo=None, hi=None, step=None, kind="float", options=None, help=None):
    d = {"name": name, "label": label, "default": default, "kind": kind}
    if help is not None:
        d["help"] = help
    if lo is not None: d["min"] = lo
    if hi is not None: d["max"] = hi
    if step is not None: d["step"] = step
    if options is not None: d["options"] = options
    return d


# ---------------- point-wise ----------------
@op("add", "Add", "value",
    [_p("value", "Amount", 0.0, -3, 3, 0.05, help="Added to every activation passing through the targeted layers.")],
    help="Shifts activations up or down. Small values brighten or darken; large values wash out structure.")
def _add(x, p, ctx):
    return x + float(p.get("value", 0.0))


@op("multiply", "Multiply", "value",
    [_p("value", "Amount", 1.0, -3, 3, 0.05, help="Activations are scaled by this factor. 1 leaves them unchanged.")],
    help="Amplifies or damps what a layer contributes. Negative values invert it.")
def _mul(x, p, ctx):
    return x * float(p.get("value", 1.0))


@op("invert", "Invert", "value", [],
    help="Flips the sign of every activation, often reversing light and dark.")
def _invert(x, p, ctx):
    return -x


@op("abs", "Absolute", "value", [],
    help="Folds negative activations to positive, collapsing contrast into ridges.")
def _abs(x, p, ctx):
    return x.abs()


@op("clamp", "Clamp", "value",
    [_p("min", "Min", -2, -5, 5, 0.1, help="Lower bound."),
     _p("max", "Max", 2, -5, 5, 0.1, help="Upper bound.")],
    help="Limits activations to a range, flattening extremes into posterised areas.")
def _clamp(x, p, ctx):
    return x.clamp(float(p.get("min", -2)), float(p.get("max", 2)))


@op("threshold", "Threshold", "value",
    [_p("t", "Amount", 0.5, 0, 3, 0.05, help="Activations weaker than this magnitude are zeroed.")],
    help="Silences weak activations, keeping only the strongest features.")
def _threshold(x, p, ctx):
    t = float(p.get("t", 0.5))
    return torch.where(x.abs() > t, x, torch.zeros_like(x))


@op("normalize", "Normalize", "value", [],
    help="Re-centres and rescales each channel, evening out layer contributions.")
def _normalize(x, p, ctx):
    m = x.mean(dim=(2, 3), keepdim=True)
    s = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
    return (x - m) / s


# ---------------- stochastic ----------------
@op("noise", "Add noise", "stochastic",
    [_p("std", "Amount", 0.2, 0, 2, 0.02, help="Standard deviation of the noise added to activations."),
     _p("seed", "Seed", 0, 0, 99999, 1, "int", help="Fixed seed so the same bend gives the same texture.")],
    help="Injects randomness mid-denoise, adding grain and breaking up smooth areas.")
def _noise(x, p, ctx):
    g = torch.Generator(device="cpu").manual_seed(int(p.get("seed", 0)))
    n = torch.randn(x.shape, generator=g).to(x.device, x.dtype)
    return x + float(p.get("std", 0.2)) * n


@op("channel_shuffle", "Channel shuffle", "stochastic",
    [_p("seed", "Seed", 0, 0, 99999, 1, "int", help="Which permutation to use.")],
    help="Reorders feature channels, usually producing strong colour shifts.")
def _cshuffle(x, p, ctx):
    g = torch.Generator(device="cpu").manual_seed(int(p.get("seed", 0)))
    perm = torch.randperm(x.shape[1], generator=g)
    return x[:, perm]


# ---------------- spatial ----------------
def _affine(x, theta):
    grid = F.affine_grid(theta.to(x.device, x.dtype), x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


@op("rotate", "Rotate", "spatial",
    [_p("angle", "Amount", 0, -180, 180, 1, help="Rotation in degrees.")],
    help="Turns the feature map, smearing structure along the rotation.")
def _rotate(x, p, ctx):
    a = math.radians(float(p.get("angle", 0)))
    theta = torch.tensor([[[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]]]).repeat(x.shape[0], 1, 1)
    return _affine(x, theta)


@op("scale", "Scale (zoom)", "spatial",
    [_p("factor", "Amount", 1.0, 0.25, 4, 0.05, help="Zoom level. 1 is unchanged, higher zooms in.")],
    help="Zooms the feature map, changing the scale of the shapes a layer produces.")
def _scale(x, p, ctx):
    f = 1.0 / max(float(p.get("factor", 1.0)), 1e-3)
    theta = torch.tensor([[[f, 0, 0], [0, f, 0]]]).repeat(x.shape[0], 1, 1)
    return _affine(x, theta)


@op("roll", "Shift (roll)", "spatial",
    [_p("dx", "Shift X", 0, -64, 64, 1, "int", help="Horizontal shift in feature pixels."),
     _p("dy", "Shift Y", 0, -64, 64, 1, "int", help="Vertical shift in feature pixels.")],
    help="Slides the feature map and wraps it around, offsetting detail from structure.")
def _roll(x, p, ctx):
    return torch.roll(x, shifts=(int(p.get("dy", 0)), int(p.get("dx", 0))), dims=(2, 3))


@op("flip", "Flip", "spatial",
    [_p("axis", "Axis", "h", options=["h", "v"], kind="select", help="h mirrors left-right, v mirrors top-bottom.")],
    help="Mirrors the feature map, often producing symmetry against the composition.")
def _flip(x, p, ctx):
    return torch.flip(x, dims=[3] if p.get("axis", "h") == "h" else [2])


# ---------------- morphological ----------------
@op("dilate", "Dilate", "morph",
    [_p("k", "Amount", 3, 1, 9, 2, "int", help="Kernel size in feature pixels (odd numbers).")],
    help="Grows bright regions, thickening strokes and filling small gaps.")
def _dilate(x, p, ctx):
    k = int(p.get("k", 3)) | 1
    return F.max_pool2d(x, k, stride=1, padding=k // 2)


@op("erode", "Erode", "morph",
    [_p("k", "Amount", 3, 1, 9, 2, "int", help="Kernel size in feature pixels (odd numbers).")],
    help="Shrinks bright regions, thinning strokes and opening up gaps.")
def _erode(x, p, ctx):
    k = int(p.get("k", 3)) | 1
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)


# ---------------- edges / frequency ----------------
def _sobel_kernels(device, dtype):
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


@op("gradient", "Gradient (Sobel)", "edges",
    [_p("mix", "Amount", 1.0, 0, 1, 0.05, help="How much edge response replaces the original activation.")],
    help="Replaces activations with their edges, giving outlined, etched results.")
def _gradient(x, p, ctx):
    kx, ky = _sobel_kernels(x.device, x.dtype)
    C = x.shape[1]
    kx = kx.repeat(C, 1, 1, 1); ky = ky.repeat(C, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    edges = torch.sqrt(gx * gx + gy * gy + 1e-6)
    mix = float(p.get("mix", 1.0))
    return (1 - mix) * x + mix * edges


@op("fourier_amplify", "Fourier amplify", "edges",
    [_p("gain", "Amount", 1.5, 0, 4, 0.1, help="How strongly high-frequency bands are boosted."),
     _p("cutoff", "Cutoff", 0.3, 0, 1, 0.05, help="Where low frequencies end and boosting starts.")],
    help="Boosts fine frequencies, sharpening texture and adding crispness.")
def _fourier(x, p, ctx):
    gain = float(p.get("gain", 1.5))
    cutoff = float(p.get("cutoff", 0.3))
    X = torch.fft.fftshift(torch.fft.fft2(x.float(), dim=(-2, -1)), dim=(-2, -1))
    _, _, H, W = X.shape
    yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1)
    xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W)
    r = torch.sqrt(xx * xx + yy * yy)
    mask = 1 + (gain - 1) * (r > cutoff).float()
    X = X * mask
    out = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), dim=(-2, -1)).real
    return out.to(x.dtype)


def apply_op(name: str, x, params: dict, ctx: dict | None = None):
    entry = OP_CATALOG.get(name)
    if entry is None:
        raise ValueError(f"unknown op: {name}")
    return entry["fn"](x, params or {}, ctx or {})


# When during a sample each category of op tends to matter.
SCHEDULE_DEFAULTS = {
    "value": {"start": 0.0, "end": 1.0},
    "stochastic": {"start": 0.0, "end": 0.45},
    "spatial": {"start": 0.15, "end": 0.85},
    "morph": {"start": 0.2, "end": 0.8},
    "edges": {"start": 0.45, "end": 1.0},
}

# Primary scalar control shown as “Amount” in the UI (None = op has no amount slider).
AMOUNT_PARAM = {
    "add": "value", "multiply": "value", "scale": "factor", "noise": "std",
    "rotate": "angle", "gradient": "mix", "fourier_amplify": "gain",
    "threshold": "t", "dilate": "k", "erode": "k",
}


def catalog() -> list[dict]:
    return [
        {
            "name": e["name"],
            "label": e["label"],
            "category": e["category"],
            "params": e["params"],
            "help": e.get("help", ""),
            "amount_param": AMOUNT_PARAM.get(e["name"]),
            "schedule": SCHEDULE_DEFAULTS.get(e["category"], SCHEDULE_DEFAULTS["value"]),
        }
        for e in OP_CATALOG.values()
    ]
