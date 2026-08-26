"""Parameter sweep runner: sample a model across a 1D or 2D range and build a contact sheet."""
import math

from PIL import Image, ImageDraw

from app.core.engine.sampler import SampleParams, sampler
from utils.imaging import build_card, data_url


def _final_frame(params: SampleParams):
    last = None
    for frame in sampler.run(params):
        last = frame
    return last["image_pp"] if last else None


def axis_values(param: str, frm, to, count: int, values=None) -> list:
    if values:
        return list(values)
    count = max(2, int(count))
    frm, to = float(frm), float(to)
    if param == "seed":
        if count == 1:
            return [int(round(frm))]
        step = (to - frm) / max(count - 1, 1)
        return [int(round(frm + i * step)) for i in range(count)]
    return [frm + (to - frm) * (i / max(count - 1, 1)) for i in range(count)]


def _fmt(param, v):
    if param == "seed" or isinstance(v, int):
        return str(int(v))
    return str(round(v, 3))


def build_contact_sheet(images: list, labels: list, cols: int | None = None) -> Image.Image:
    imgs = [im for im in images if im is not None]
    if not imgs:
        raise ValueError("no images produced")
    n = len(imgs)
    cols = cols or min(n, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    cw, ch = imgs[0].size
    pad, labelh = 6, 18
    W = cols * cw + (cols + 1) * pad
    H = rows * (ch + labelh) + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (20, 22, 28))
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x = pad + c * (cw + pad)
        y = pad + r * (ch + labelh + pad)
        sheet.paste(im.resize((cw, ch)), (x, y))
        draw.text((x + 2, y + ch + 3), str(labels[i]), fill=(230, 233, 238))
    return sheet


def run_sweep(job, model_path: str, axes: list, base: dict):
    """axes: list of {param, values} length 1 or 2."""
    ax0 = axes[0]
    xs = ax0["values"]
    ys = axes[1]["values"] if len(axes) > 1 else [None]
    p0, p1 = ax0["param"], (axes[1]["param"] if len(axes) > 1 else None)
    cols = len(xs)
    total = len(xs) * len(ys)
    images, labels, cells = [], [], []
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
            kwargs[p0] = int(x) if p0 == "seed" else x
            if p1 is not None:
                kwargs[p1] = int(y) if p1 == "seed" else y
            params = SampleParams(model_path=model_path, **kwargs)
            img = _final_frame(params)
            lab = f"{p0}={_fmt(p0, x)}" + (f" · {p1}={_fmt(p1, y)}" if p1 else "")
            images.append(img)
            labels.append(lab)
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
    sheet = build_contact_sheet(images, labels, cols=cols)
    return sheet
