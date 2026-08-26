"""Masked img2img (regional fill) on top of the xurdif DDIM sampler.

xurdif has no inpaint UNet / mask channel. We composite each step: denoise inside
the (feathered) mask and keep the init image outside. Same honesty as RePaint-style
pixel-space compositing.

Region fill runs at the *canvas's* resolution, not the Sample-settings one: the
sampler returns the whole frame, so anything else would resample every unmasked
pixel and change the canvas's aspect.
"""
from PIL import Image, ImageFilter

from app.core.engine.sampler import SampleParams, align_size, sampler

# Beyond this the run drifts far from any plausible training resolution and a
# 6 GB card starts to struggle; we sample smaller and scale the result back.
MAX_FILL_SIDE = 1024


def fill_size(canvas: Image.Image, mults, max_side: int = MAX_FILL_SIDE) -> tuple[int, int]:
    """The (width, height) region fill should sample at for this canvas."""
    w, h = canvas.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        w, h = round(w * scale), round(h * scale)
    return align_size(w, mults), align_size(h, mults)


def prepare_mask(mask: Image.Image, size: tuple[int, int], feather: float = 8.0) -> Image.Image:
    m = mask.convert("L").resize(size, Image.BILINEAR)
    if feather and feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return m


def run_inpaint(
    params: SampleParams,
    init_image: Image.Image,
    mask: Image.Image,
    *,
    feather: float = 8.0,
    image_prompt=None,
    bend_runtime=None,
    cancel=None,
    control=None,
    mults=None,
):
    if init_image is None:
        raise ValueError("inpaint needs an init image (the canvas)")

    out_size = init_image.size                      # what the canvas expects back
    w, h = fill_size(init_image, mults)             # what the UNet can process
    prepared = prepare_mask(mask, (w, h), feather)

    # The sampler composites in latent/tensor space, but `_to_images` then applies
    # a per-sample contrast normalisation across the *whole* frame — which shifts
    # every untouched pixel, so repeated fills would slowly drift the canvas. Do a
    # final composite in pixel space against the original so anything outside the
    # mask comes back bit-exact.
    base = init_image.convert("RGB")
    out_mask = prepared.resize(out_size, Image.BILINEAR) if (w, h) != out_size else prepared

    def _merge(img):
        return Image.composite(img.convert("RGB").resize(out_size, Image.LANCZOS), base, out_mask)

    for frame in sampler.run(
        params,
        init_image=init_image,
        image_prompt=image_prompt,
        bend_runtime=bend_runtime,
        cancel=cancel,
        mask=prepared,
        control=control,
        height=h,
        width=w,
    ):
        for key in ("image", "image_pp"):
            if frame.get(key) is not None:
                frame[key] = _merge(frame[key])
        for key in ("images", "images_pp"):
            if frame.get(key):
                frame[key] = [_merge(im) for im in frame[key]]
        yield frame
