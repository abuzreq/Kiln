"""Resize / crop / pad images into a consistent training size."""
from PIL import Image, ImageOps

RESIZE_MODES = ("stretch", "center_crop", "pad")
PADDING_MODES = ("edge", "reflect", "constant")


def process_image(
    img: Image.Image,
    width: int,
    height: int,
    mode: str = "center_crop",
    padding_mode: str = "edge",
    pad_color: tuple = (0, 0, 0),
) -> Image.Image:
    img = img.convert("RGB")
    if mode == "stretch":
        return img.resize((width, height), Image.LANCZOS)

    if mode == "center_crop":
        return ImageOps.fit(img, (width, height), Image.LANCZOS, centering=(0.5, 0.5))

    if mode == "pad":
        src_w, src_h = img.size
        scale = min(width / src_w, height / src_h)
        new_w, new_h = max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        if padding_mode == "constant":
            canvas = Image.new("RGB", (width, height), pad_color)
            canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
            return canvas
        # edge / reflect padding via numpy
        import numpy as np

        arr = np.array(resized)
        pad_l = (width - new_w) // 2
        pad_r = width - new_w - pad_l
        pad_t = (height - new_h) // 2
        pad_b = height - new_h - pad_t
        np_mode = "edge" if padding_mode == "edge" else "reflect"
        padded = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=np_mode)
        return Image.fromarray(padded)

    raise ValueError(f"unknown resize mode: {mode}")
