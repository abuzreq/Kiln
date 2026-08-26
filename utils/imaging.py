"""Base64 <-> PIL helpers for passing images over JSON."""
import base64
import io

from PIL import Image


def data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def preview_url(img: Image.Image, max_side: int = 384, quality: int = 80) -> str:
    """A small JPEG data URL for live step previews.

    Encoding a full-resolution PNG on every denoise step cost ~39 ms each (and it
    was done twice), which dwarfed the ~32 ms UNet forward. Previews are a
    thumbnail the user watches, not the output, so they get downscaled JPEG; the
    real PNG is emitted once when the run finishes.
    """
    im = img
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side), Image.BILINEAR)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=False)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def from_data_url(s: str) -> Image.Image:
    if not s:
        raise ValueError("empty image")
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def from_data_url_mask(s: str) -> Image.Image:
    """Decode a data URL as a grayscale mask (L)."""
    if not s:
        raise ValueError("empty mask")
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    return Image.open(io.BytesIO(raw)).convert("L")


# --- embedded generation metadata -------------------------------------
#
# Kiln writes the full sampling recipe into the PNG itself so an image is
# self-describing: drop it back on the canvas (in any later session, or on
# another machine) and the run that made it can be reconstructed. Two forms are
# written side by side:
#
#   "kiln"       iTXt  — the authoritative JSON card (read back by Kiln)
#   "parameters" tEXt  — a human-readable one-liner other tools already display
#   "Software"   tEXt  — provenance
KILN_CHUNK = "kiln"
CARD_VERSION = 1

# Sampling fields copied verbatim from a SampleParams-shaped payload.
_PARAM_KEYS = (
    "image_size", "steps", "eta", "skip", "seed", "batch_size",
    "text", "text_weight", "guidance_step", "guidance_power",
    "image_prompt_weight", "cuts", "noise_level", "ema", "sampler",
    "resample", "jump_length",
)


def build_card(params, *, model_path=None, model_name=None, bends=None,
               bend_preset=None, postproc=None, init_image=False, mask=False,
               kind="sample", extra=None) -> dict:
    """Assemble the metadata card stored in a generated PNG.

    ``params`` may be a SampleParams dataclass or a plain dict.
    """
    import time
    from pathlib import Path

    from app import __version__

    get = params.get if isinstance(params, dict) else lambda k, d=None: getattr(params, k, d)
    card = {
        "version": CARD_VERSION,
        "app": "kiln",
        "app_version": __version__,
        "kind": kind,
        "created_at": time.time(),
        "params": {k: get(k) for k in _PARAM_KEYS},
        "init_image": bool(init_image),
        "mask": bool(mask),
    }
    path = model_path if model_path is not None else get("model_path")
    if path:
        card["model_path"] = str(path)
        card["model"] = model_name or Path(str(path)).stem
    if bends:
        card["bends"] = bends
    if bend_preset:
        card["bend_preset"] = bend_preset
    if postproc:
        card["postproc"] = postproc
    if extra:
        card.update(extra)
    return card


def _summary_line(card: dict) -> str:
    p = card.get("params") or {}
    bits = []
    if p.get("steps") is not None:
        bits.append(f"Steps: {p['steps']}")
    if p.get("seed") is not None:
        bits.append(f"Seed: {p['seed']}")
    if p.get("eta") is not None:
        bits.append(f"Eta: {p['eta']}")
    if p.get("image_size"):
        bits.append(f"Size: {p['image_size']}x{p['image_size']}")
    if card.get("model"):
        bits.append(f"Model: {card['model']}")
    if card.get("bend_preset"):
        bits.append(f"Bend: {card['bend_preset']}")
    elif card.get("bends"):
        bits.append(f"Bend: {len(card['bends'])} op(s)")
    bits.append(f"Sampler: {p.get('sampler') or 'DDIM'}")
    head = (p.get("text") or "").strip()
    return (head + "\n" if head else "") + ", ".join(bits)


def png_info(card: dict):
    """Build a PngInfo carrying ``card`` plus human-readable fallbacks."""
    import json

    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_itxt(KILN_CHUNK, json.dumps(card, separators=(",", ":")))
    info.add_text("parameters", _summary_line(card))
    info.add_text("Software", f"Kiln {card.get('app_version', '')}".strip())
    return info


def save_with_params(img: Image.Image, path, card: dict | None = None):
    """Save a PNG with the generation recipe embedded. Returns ``path``."""
    if card:
        img.save(str(path), format="PNG", pnginfo=png_info(card))
    else:
        img.save(str(path), format="PNG")
    return path


def bytes_with_params(img: Image.Image, card: dict | None = None) -> bytes:
    """PNG bytes with the recipe embedded — for streaming a download."""
    buf = io.BytesIO()
    if card:
        img.save(buf, format="PNG", pnginfo=png_info(card))
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def read_params(src) -> dict | None:
    """Read a Kiln card back out of a PNG (path, file object, or PIL image).

    Falls back to the plain ``parameters`` text when the JSON chunk is absent,
    so images from other tools still yield something useful.
    """
    import json

    img = src if isinstance(src, Image.Image) else Image.open(src)
    info = getattr(img, "info", None) or {}
    raw = info.get(KILN_CHUNK)
    if raw:
        try:
            card = json.loads(raw)
            if isinstance(card, dict):
                return card
        except (ValueError, TypeError):
            pass
    text = info.get("parameters")
    if text:
        return {"version": 0, "app": info.get("Software", ""), "parameters": text}
    return None
