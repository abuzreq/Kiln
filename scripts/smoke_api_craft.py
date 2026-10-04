"""End-to-end API smoke for craft + library: place a model, introspect, preview, save."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import _scratch_workspace  # noqa: E402,F401  -- before app: never write to ~/kiln

import torch
from app.core.engine.arch import build_unet
from app.core.config import workspace
from app.backend.app import create_app

# place a synthetic model in the workspace
mtype, mults = "tinyunet_with_attention3", [1, 2, 2, 2]
unet = build_unet(mtype, mults)
state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
mp = workspace.models / "demo.pt"
torch.save({"step": 0, "model": state, "ema": state, "mults": mults, "mtype": mtype, "pred": "x0"}, mp)

c = create_app().test_client()
models = c.get("/api/models").get_json()["data"]
print("models listed:", [m["name"] for m in models])
path = next(m["path"] for m in models if m["name"] == "demo")

g = c.post("/api/craft/introspect", json={"model_path": path}).get_json()["data"]
print("introspect nodes:", len(g["nodes"]))

ops = c.get("/api/craft/ops").get_json()["data"]
print("ops:", len(ops["ops"]), "groups:", ops["groups"])

bends = [{"op": "rotate", "params": {"angle": 45}, "targets": ["mid"], "step_start": 0, "step_end": 1, "active": True}]

# the layer activation preview was removed — the route should be gone with it
gone = c.post("/api/craft/preview", json={"model_path": path, "node_id": "mid_block1", "bends": bends})
assert gone.status_code == 404, f"/craft/preview still routed ({gone.status_code})"
print("removed /craft/preview returns 404")

sv = c.post("/api/craft/bends", json={"name": "test-bend", "bends": bends}).get_json()
print("saved bend ok:", sv["ok"])
lib = c.get("/api/library/bends").get_json()["data"]
print("library bends:", [b["name"] for b in lib])

# a preset keeps the picture it was saved with; anything else offered as one is dropped
thumb = "data:image/jpeg;base64," + "A" * 64
c.post("/api/craft/bends", json={"name": "with-thumb", "bends": bends, "thumbnail": thumb})
c.post("/api/craft/bends", json={"name": "bad-thumb", "bends": bends,
                                  "thumbnail": "javascript:alert(1)"})
c.post("/api/craft/bends", json={"name": "huge-thumb", "bends": bends,
                                  "thumbnail": "data:image/png;base64," + "A" * 200_000})
by_name = {b["name"]: b for b in c.get("/api/craft/bends").get_json()["data"]}
assert by_name["with-thumb"].get("thumbnail") == thumb, "a preset lost its picture"
assert "thumbnail" not in by_name["bad-thumb"], "a non-image thumbnail was stored"
assert "thumbnail" not in by_name["huge-thumb"], "an oversized thumbnail was stored"
print("preset thumbnails: kept when an image, dropped otherwise")

# --- bend sweep: full samples varying one bend parameter -----------------
import time as _t

nodes = g["nodes"]
node = nodes[len(nodes) // 2]["id"]
stack = [{"op": "noise", "params": {"std": 0.0, "seed": 0}, "targets": [node],
          "step_start": 0, "step_end": 1, "active": True}]
r = c.post("/api/craft/bend/sweep", json={
    "model_path": path, "bends": stack, "bend_index": 0, "param": "std",
    "from": 0.0, "to": 1.5, "count": 3, "image_size": 64, "steps": 4,
    "seed": 11, "sampler": "ddim",
}).get_json()
assert r.get("ok") is not False, r
job = r["data"]["job"]
for _ in range(400):
    job = c.get(f"/api/jobs/{job['id']}").get_json()["data"]
    if job["status"] in ("done", "error", "cancelled"):
        break
    _t.sleep(0.25)
assert job["status"] == "done", job["message"]
frames = job["detail"]["frames"]
assert len(frames) == 3, frames
assert [f["value"] for f in frames] == [0.0, 0.75, 1.5], frames
# the bend must actually move the picture, or the sweep is pointless
assert frames[0]["image"] != frames[-1]["image"], "bend sweep produced identical frames"
assert job["detail"]["card"]["bend_param"] == "std"
print("bend sweep:", [f["value"] for f in frames], "-> frames differ")

# a bad parameter name is rejected rather than silently swept
bad = c.post("/api/craft/bend/sweep", json={
    "model_path": path, "bends": stack, "bend_index": 0, "param": "nope",
    "from": 0, "to": 1, "count": 2,
})
assert bad.status_code == 400, bad.status_code
print("bend sweep rejects unknown parameters")

# --- GIF assembly ---------------------------------------------------------
gif = c.post("/api/tools/gif", json={
    "images": [f["image"] for f in frames], "fps": 6,
    "pingpong": True, "name": "smoke_bendgif",
}).get_json()["data"]
assert gif["frames"] == 4, gif          # 3 forward + 1 coming back
from PIL import Image as _Im

with _Im.open(gif["path"]) as _g:       # close the handle before unlinking
    assert getattr(_g, "is_animated", False) and _g.n_frames == 4
    print("gif:", gif["frames"], "frames,", _g.size)
Path(gif["path"]).unlink(missing_ok=True)

single = c.post("/api/tools/gif", json={"images": [frames[0]["image"]]})
assert single.status_code == 400, "a one-frame GIF should be refused"
print("gif needs at least two frames")

# cleanup
mp.unlink(missing_ok=True)
(workspace.bends / "test-bend.json").unlink(missing_ok=True)
print("OK")
