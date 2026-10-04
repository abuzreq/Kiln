"""Parameter sweep runner: sample a model across a 1D or 2D range and build a contact sheet."""

from PIL import Image, ImageDraw

from app.core.engine.sampler import SampleParams, sampler
from utils.imaging import build_card, data_url


def _final_frame(params: SampleParams):
    last = None
    for frame in sampler.run(params):
        last = frame
    return last["image_pp"] if last else None


def axis_values(param: str, frm, to, count: int, values=None) -> list:
    """Values for one axis. An explicit list wins -- geometric and categorical
    axes are laid out by the caller, since they are not a line between numbers."""
    if values:
        return [_coerce(param, v) for v in values]
    count = max(2, int(count))
    frm, to = float(frm), float(to)
    span = [frm + (to - frm) * (i / max(count - 1, 1)) for i in range(count)]
    if param in INT_AXES:
        return [int(round(v)) for v in span]
    return span


# Axes whose values must reach SampleParams as ints, not floats.
INT_AXES = ("seed", "steps", "image_size", "skip", "batch_size", "resample")


def _fmt(param, v):
    """Label one axis value. Not every axis is a number -- sampler is a name."""
    if isinstance(v, str):
        return v
    if param in INT_AXES or isinstance(v, int):
        return str(int(v))
    return str(round(v, 3))


def _coerce(param, v):
    return int(v) if (param in INT_AXES and not isinstance(v, str)) else v


# What a sheet calls each axis, as the Sweep tab names them (paramLabel).
AXIS_NAMES = {"seed": "Seed", "steps": "Steps", "sampler": "Sampler", "image_size": "Image size"}


def _axis_value(param, v) -> str:
    if param == "sampler":
        from app.core.engine.sampler import SAMPLERS

        return SAMPLERS.get(v, {}).get("label", str(v)).split(" (")[0]
    if param == "image_size":
        return f"{_fmt(param, v)}px"
    return _fmt(param, v)


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):  # Pillow without FreeType: the fixed bitmap font
        return ImageFont.load_default()


def build_axes_sheet(images: list, x_axis: dict, y_axis: dict | None = None) -> Image.Image:
    """A grid read like a chart: x values across the top, y values down the left,
    each axis named. ``images`` run row by row; axes are ``{param, values}``."""
    ys = y_axis["values"] if y_axis else [None]
    xs = x_axis["values"]
    cw, ch = next(im for im in images if im is not None).size
    pad = max(6, cw // 64)
    fs = max(14, cw // 16)               # still legible once a wide sheet is shrunk
    font, title = _font(fs), _font(round(fs * 0.85))
    dim, bright = (154, 161, 177), (231, 233, 238)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def width(text, f):
        return probe.textbbox((0, 0), text, font=f)[2]

    head = round(fs * 2.9)               # axis name, then the column values
    left = 0
    if y_axis:
        left = max(width(_axis_value(y_axis["param"], y), font) for y in ys)
        left = max(left, width(AXIS_NAMES.get(y_axis["param"], y_axis["param"]), title)) + 3 * pad
    W = left + len(xs) * cw + (len(xs) + 1) * pad
    H = head + len(ys) * ch + (len(ys) + 1) * pad
    sheet = Image.new("RGB", (W, H), (20, 22, 28))
    draw = ImageDraw.Draw(sheet)

    x_name = AXIS_NAMES.get(x_axis["param"], x_axis["param"])
    grid_left = left + pad
    grid_w = len(xs) * cw + (len(xs) - 1) * pad
    draw.text((grid_left + grid_w / 2, pad), x_name, font=title, fill=dim, anchor="mt")
    for c, x in enumerate(xs):
        cx = grid_left + c * (cw + pad) + cw / 2
        draw.text((cx, head - pad // 2), _axis_value(x_axis["param"], x), font=font,
                  fill=bright, anchor="mb")
    if y_axis:
        y_name = AXIS_NAMES.get(y_axis["param"], y_axis["param"])
        draw.text((pad, pad), y_name, font=title, fill=dim, anchor="lt")
    for i, im in enumerate(images):
        if im is None:
            continue
        r, c = divmod(i, len(xs))
        x0 = grid_left + c * (cw + pad)
        y0 = head + pad + r * (ch + pad)
        sheet.paste(im.resize((cw, ch)), (x0, y0))
        if y_axis and c == 0:
            draw.text((left, y0 + ch / 2), _axis_value(y_axis["param"], ys[r]), font=font,
                      fill=bright, anchor="rm")
    return sheet


def run_sweep(job, model_path: str, axes: list, base: dict):
    """axes: list of {param, values} length 1 or 2."""
    ax0 = axes[0]
    xs = ax0["values"]
    ys = axes[1]["values"] if len(axes) > 1 else [None]
    p0, p1 = ax0["param"], (axes[1]["param"] if len(axes) > 1 else None)
    cols = len(xs)
    total = len(xs) * len(ys)
    images, cells = [], []
    if job is not None:
        job.detail["cols"] = cols
        job.detail["rows"] = len(ys)
        job.detail["axis_x"] = {"param": p0, "values": xs}
        job.detail["axis_y"] = {"param": p1, "values": ys} if p1 else None
        job.detail["cells"] = []
        job.detail["planned"] = [
            f"{p0}={_fmt(p0, x)}" + (f" · {p1}={_fmt(p1, y)}" if p1 else "")
            for y in ys for x in xs
        ]

    n_done = 0
    for y in ys:
        for x in xs:
            if job is not None and job.cancelled():
                job.status = "cancelled"
                return None
            kwargs = dict(base)
            kwargs[p0] = _coerce(p0, x)
            if p1 is not None:
                kwargs[p1] = _coerce(p1, y)
            params = SampleParams(model_path=model_path, **kwargs)
            img = _final_frame(params)
            lab = f"{p0}={_fmt(p0, x)}" + (f" · {p1}={_fmt(p1, y)}" if p1 else "")
            images.append(img)
            cell = {
                "label": lab, "x": x, "y": y,
                "image": data_url(img) if img else None,
                # each cell is a complete run — carry its recipe so a good one
                # can be sent straight to the canvas without retyping settings
                "card": build_card(params, kind="sweep-cell", postproc=params.postproc),
            }
            cells.append(cell)
            n_done += 1
            if job is not None:
                job.progress = n_done / total
                job.message = f"sampled {n_done}/{total} ({lab})"
                job.detail["cells"] = cells
    # Axes on the sheet itself, so it reads without the page: a label under
    # every cell said "sampler=dpmpp · seed=0" sixteen times over.
    return build_axes_sheet(
        images,
        {"param": p0, "values": xs},
        {"param": p1, "values": ys} if p1 else None,
    )
