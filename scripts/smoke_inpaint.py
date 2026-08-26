"""Region-fill smoke: the fill runs at the canvas's size, not the Sample-settings one.

Regression guard for the bug where `H = W = params.image_size` forced every
masked run to a square at whatever `Image size` happened to be — squashing a
non-square canvas and resampling every unmasked pixel on the way through.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageChops
from app.core.engine.arch import build_unet
from app.core.engine.inpaint import fill_size
from app.core.engine.sampler import align_size
from app.core.config import workspace
from app.backend.app import create_app
from utils.imaging import data_url

unet = build_unet("tinyunet_with_attention3", [1, 2, 2, 2])
state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
mp = workspace.models / "inpainttest.pt"
torch.save({"step": 0, "model": state, "ema": state, "mults": [1, 2, 2, 2],
            "mtype": "tinyunet_with_attention3", "pred": "x0"}, mp)

app = create_app()
c = app.test_client()

# --- the alignment rule the UNet actually imposes -------------------------
assert align_size(900, [1, 2, 2, 2]) == 896, "must snap to a multiple of 2**len(mults)"
assert align_size(900, [1, 2, 2, 2, 2]) == 896
assert align_size(10, [1, 2, 2, 2]) == 16, "never smaller than one step"
assert fill_size(Image.new("RGB", (1920, 1080)), [1, 2, 2, 2])[0] == 1024, "long side is capped"
print("size math OK")

# --- a deliberately non-square canvas, with Image size set to something else ---
CANVAS = (208, 144)          # not square, not a multiple of 16 (-> 208x144 is, on purpose mixed)
canvas = Image.new("RGB", CANVAS, (40, 90, 140))
for x in range(CANVAS[0]):   # gradient so any resample round-trip is visible
    for y in range(0, CANVAS[1], 8):
        canvas.putpixel((x, y), (x % 256, 200, 60))

mask = Image.new("L", CANVAS, 0)
for x in range(20, 70):      # paint a small region on the left
    for y in range(20, 70):
        mask.putpixel((x, y), 255)

def fill(**over):
    body = {
        "model_path": str(mp),
        "init_image": data_url(canvas),
        "mask": data_url(mask.convert("RGB")),
        "image_size": 64,        # deliberately WRONG for the canvas — must be ignored
        "steps": 4,
        "feather": 0,
        **over,
    }
    j = c.post("/api/perform/inpaint", json=body).get_json()["data"]["job"]
    for _ in range(400):
        j = c.get(f"/api/jobs/{j['id']}").get_json()["data"]
        if j["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.25)
    return j


# Every solver, with and without a skip. The multistep solvers iterate their
# timesteps inside add_noise, so a 0-d timestep used to raise "iteration over a
# 0-d tensor" here — a path DDIM alone never exercised.
from app.core.engine.sampler import SAMPLERS

for name in SAMPLERS:
    for skip in (0, 2):
        j = fill(sampler=name, skip=skip, noise_level=0.74)
        assert j["status"] == "done", f"{name} skip={skip}: {j['message']}"
        from utils.imaging import from_data_url as _fd
        assert _fd(j["detail"]["frame"]).size == CANVAS, f"{name} returned the wrong size"
print(f"all {len(SAMPLERS)} samplers fill a masked init image, with and without skip")

job = fill()
assert job["status"] == "done", job["message"]

# the route reports what it actually sampled at
assert job["detail"]["canvas_size"] == list(CANVAS), job["detail"]
assert job["detail"]["fill_size"] == list(fill_size(canvas, [1, 2, 2, 2])), job["detail"]
print("reported fill size:", job["detail"]["fill_size"], "for canvas", job["detail"]["canvas_size"])

# --- the returned frame must match the canvas exactly, not image_size ---
from utils.imaging import from_data_url

out = from_data_url(job["detail"]["frame"])
assert out.size == CANVAS, f"expected {CANVAS}, got {out.size} (image_size leaked back in)"
print("output size matches the canvas:", out.size)

# --- unmasked pixels must survive untouched ---
diff = ImageChops.difference(out.convert("RGB"), canvas)
unmasked_changed = 0
for x in range(CANVAS[0]):
    for y in range(CANVAS[1]):
        if mask.getpixel((x, y)) == 0:
            r, g, b = diff.getpixel((x, y))
            if max(r, g, b) > 24:
                unmasked_changed += 1
total_unmasked = sum(1 for x in range(CANVAS[0]) for y in range(CANVAS[1]) if mask.getpixel((x, y)) == 0)
pct = 100.0 * unmasked_changed / max(total_unmasked, 1)
print(f"unmasked pixels materially changed: {unmasked_changed}/{total_unmasked} ({pct:.1f}%)")
assert pct < 5.0, "region fill is disturbing pixels outside the mask"

mp.unlink(missing_ok=True)
print("OK")
