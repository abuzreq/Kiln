"""Masked img2img (regional fill).

Neither engine has an inpaint UNet or a mask channel, so the mask is applied by
compositing: denoise inside the (feathered) mask, keep the init image outside.
The denoise loop itself lives in ``sampler.Sampler.run`` — including RePaint
resampling — so there is exactly one implementation of masked compositing.

**Relationship to RePaint** (Lugmayr et al., arXiv:2201.09865). Kiln's default
fill already implements RePaint's *compositing* faithfully: each step re-noises
the known region to the next timestep with
``√ᾱ₍ₜ₋₁₎·x₀ + √(1-ᾱ₍ₜ₋₁₎)·ε``, which is the paper's ``prev_known_part``. What it
does not do by default is RePaint's *resampling* — repeatedly jumping back up
the schedule so the model can reconcile the generated region with its
surroundings. Without it the known region is re-noised "without considering the
generated parts", so a fill lines up at the seam without necessarily agreeing
with the content around it. ``SampleParams.resample`` turns it on; it is off by
default because it costs several times more model evaluations.

**Three things Kiln does that RePaint does not**, all deliberate:

1. **A feathered mask.** RePaint's mask is binary; Kiln blurs it, so the mask's
   gray values act as a per-pixel blend and a fill does not leave a hard edge.
2. **A final composite in pixel space.** The sampler composites in tensor space,
   but ``Backend.to_display`` then renormalises contrast across the *whole*
   frame, which shifts every untouched pixel — repeated fills would slowly drift
   the canvas. The composite against the original below makes anything outside
   the mask come back bit-exact.
3. **Fills at the canvas's own resolution**, snapped to the model's
   ``size_multiple``, rather than the square Sample-settings size. The sampler
   returns a whole frame, so anything else would resample every unmasked pixel
   and change the canvas's aspect.
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

    # See note 2 in the module docstring: this is what keeps unmasked pixels
    # bit-exact across repeated fills.
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
        # Deferred with the render: a step nobody looks at is never composited.
        yield frame.map_images(_merge)
