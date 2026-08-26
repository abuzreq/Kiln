"""Full-stack API smoke: every blueprint registers; tools (superres/sweep/capture) work."""
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
from utils.imaging import data_url

# synthetic model
unet = build_unet("tinyunet_with_attention3", [1, 2, 2, 2])
state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
mp = workspace.models / "fulltest.pt"
torch.save({"step": 0, "model": state, "ema": state, "mults": [1, 2, 2, 2],
            "mtype": "tinyunet_with_attention3", "pred": "x0"}, mp)

app = create_app()
c = app.test_client()

# every optional blueprint must be present -> these routes exist
for path in ["/api/health", "/api/models", "/api/craft/ops", "/api/library/catalog", "/api/captures"]:
    assert c.get(path).status_code == 200, path
print("all core routes respond")

# super-res
img = Image.new("RGB", (32, 32), (120, 60, 200))
r = c.post("/api/tools/superres", json={"image": data_url(img), "factor": 2, "sharpen": 0.5}).get_json()["data"]
print("superres ->", r["size"])

# capture + list
cap = c.post("/api/perform/capture", json={"image": data_url(img), "name": "smoke_cap"}).get_json()
print("capture saved:", cap["ok"])
print("captures listed:", any("smoke_cap" in f for f in c.get("/api/captures").get_json()["data"]["files"]))

# sweep (tiny, CPU)
sj = c.post("/api/tools/sweep", json={"model_path": str(mp), "param": "seed", "count": 2,
                                      "image_size": 48, "steps": 3}).get_json()["data"]["job"]
for _ in range(120):
    job = c.get(f"/api/jobs/{sj['id']}").get_json()["data"]
    if job["status"] in ("done", "error", "cancelled"):
        break
    time.sleep(0.5)
print("sweep status:", job["status"], "| has sheet:", bool(job["detail"].get("sheet")))

# cleanup
mp.unlink(missing_ok=True)
(workspace.captures / "smoke_cap.png").unlink(missing_ok=True)
for f in workspace.captures.glob("sweep_seed_*.png"):
    f.unlink(missing_ok=True)
print("OK")
