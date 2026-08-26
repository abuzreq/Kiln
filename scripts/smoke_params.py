"""Reproducibility smoke: every generated image carries the recipe that made it.

Covers the round trip end to end — a blank seed is resolved and reported, the
recipe is embedded in captured and exported PNGs, it reads back out of a data
URL, and re-running the reported seed reproduces the image exactly.
"""
import base64
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from app.core.engine.arch import build_unet
from app.core.config import workspace
from app.backend.app import create_app
from utils.imaging import data_url, read_params

unet = build_unet("tinyunet_with_attention3", [1, 2, 2, 2])
state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
mp = workspace.models / "paramtest.pt"
torch.save({"step": 0, "model": state, "ema": state, "mults": [1, 2, 2, 2],
            "mtype": "tinyunet_with_attention3", "pred": "x0"}, mp)

app = create_app()
c = app.test_client()
SMALL = {"image_size": 48, "steps": 3}


def run_sample(**over):
    body = {"model_path": str(mp), **SMALL, **over}
    job = c.post("/api/perform/sample", json=body).get_json()["data"]["job"]
    for _ in range(240):
        job = c.get(f"/api/jobs/{job['id']}").get_json()["data"]
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.25)
    raise AssertionError("sample did not finish")


# --- a blank seed must still be a concrete, reported seed -----------------
job = run_sample(seed="", bend_preset="smoke-preset")
assert job["status"] == "done", job["message"]
seed = job["detail"].get("seed")
assert isinstance(seed, int), f"no resolved seed: {seed!r}"
card = job["detail"].get("card")
assert card and card["params"]["seed"] == seed, card
assert card["bend_preset"] == "smoke-preset"
print("blank seed resolved ->", seed)

# --- that seed reproduces the image ---------------------------------------
again = run_sample(seed=seed)
assert again["detail"]["frame"] == job["detail"]["frame"], "same seed did not reproduce"
print("re-run with reported seed: identical")

# --- capture embeds the recipe in the file on disk -------------------------
cap = c.post("/api/perform/capture", json={
    "image": job["detail"]["frame"], "name": "smoke_params", "card": card,
}).get_json()["data"]
on_disk = read_params(cap["path"])
assert on_disk["params"]["seed"] == seed, on_disk
assert on_disk["model"] == "paramtest", on_disk
print("capture PNG carries recipe:", Path(cap["path"]).name)

# --- and so does a download ------------------------------------------------
res = c.post("/api/perform/export", json={
    "image": job["detail"]["frame"], "card": card, "filename": "smoke_dl",
})
assert res.status_code == 200 and res.headers["Content-Type"] == "image/png"
assert read_params(Image.open(io.BytesIO(res.data)))["params"]["seed"] == seed
print("exported PNG carries recipe")

# --- reading it back out of a dropped image --------------------------------
# The browser's FileReader.readAsDataURL sends the file's raw bytes, so the PNG
# chunks survive; re-encoding through PIL would silently drop them.
def file_data_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


back = c.post("/api/perform/read-params", json={
    "image": file_data_url(cap["path"]),
}).get_json()["data"]["card"]
assert back["params"]["seed"] == seed, back
assert c.post("/api/perform/read-params", json={
    "image": data_url(Image.new("RGB", (8, 8))),
}).get_json()["data"]["card"] is None
print("read-params round-trips; plain images return null")

# --- captures listing exposes the recipe -----------------------------------
listed = c.get("/api/captures").get_json()["data"]["captures"]
mine = next(e for e in listed if e["name"] == "smoke_params")
assert mine["card"]["params"]["seed"] == seed
print("captures listing exposes recipe")

# --- the solver is part of the recipe --------------------------------------
uni = run_sample(seed=4242, sampler="unipc")
assert uni["status"] == "done", uni["message"]
assert uni["detail"]["card"]["params"]["sampler"] == "unipc", uni["detail"]["card"]["params"]
ddim = run_sample(seed=4242, sampler="ddim")
assert ddim["detail"]["card"]["params"]["sampler"] == "ddim"
# different solvers walk different trajectories, so the same seed must differ
assert uni["detail"]["frame"] != ddim["detail"]["frame"], "sampler choice had no effect"
# and a solver reproduces itself
again = run_sample(seed=4242, sampler="unipc")
assert again["detail"]["frame"] == uni["detail"]["frame"], "unipc is not deterministic"
# omitting it must stay DDIM, so cards written before the field replay correctly
legacy = run_sample(seed=4242)
assert legacy["detail"]["card"]["params"]["sampler"] == "ddim", "default drifted away from DDIM"
assert legacy["detail"]["frame"] == ddim["detail"]["frame"]
print("sampler recorded, deterministic, and defaults to DDIM for old cards")

# --- sweep: every cell is its own reproducible run -------------------------
sj = c.post("/api/tools/sweep", json={
    "model_path": str(mp), "param": "seed", "from": 1, "to": 2, "count": 2, **SMALL,
}).get_json()["data"]["job"]
for _ in range(240):
    sweep = c.get(f"/api/jobs/{sj['id']}").get_json()["data"]
    if sweep["status"] in ("done", "error", "cancelled"):
        break
    time.sleep(0.25)
assert sweep["status"] == "done", sweep["message"]
cells = sweep["detail"]["cells"]
assert all(cell["card"]["params"]["seed"] is not None for cell in cells), "cell without a seed"
assert cells[0]["card"]["params"]["seed"] != cells[1]["card"]["params"]["seed"]
sheet_card = read_params(sweep["detail"]["path"])
assert sheet_card["axes"][0]["param"] == "seed", sheet_card
print("sweep: per-cell recipes +", len(sheet_card["axes"]), "axis recorded on the sheet")

# cleanup
mp.unlink(missing_ok=True)
c.delete("/api/captures", json={"path": cap["path"]})
Path(sweep["detail"]["path"]).unlink(missing_ok=True)
print("OK")
