"""API smoke for generations going through the lanes (CPU, untrained models).

Covers what the routes add on top of scripts/smoke_lanes.py: jobs start
queued, cancelling a queued job, the light job list, runs on two models at
once, a bad model failing as a job, and saving into a captures folder (what
Create does with a queue's results).
"""
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import _scratch_workspace  # noqa: E402,F401  -- before app: never write to ~/kiln

import torch  # noqa: E402

from app.backend.app import create_app  # noqa: E402
from app.core.config import workspace  # noqa: E402
from app.core.engine.arch import build_unet  # noqa: E402
from utils.imaging import read_params  # noqa: E402

TERMINAL = ("done", "error", "cancelled")
FAST = {"image_size": 64, "steps": 2, "device": "cpu", "sampler": "ddim"}
# Long enough to still be running when the next request lands; always cancelled.
SLOW = {**FAST, "steps": 1000}


def place_model(name):
    mtype, mults = "tinyunet_with_attention3", [1, 2, 2, 2]
    state = {f"denoise_fn.{k}": v for k, v in build_unet(mtype, mults).state_dict().items()}
    path = workspace.models / f"{name}.pt"
    torch.save({"step": 0, "model": state, "ema": state, "mults": mults, "mtype": mtype,
                "pred": "x0"}, path)
    return path


def main():
    a, b = place_model("smoke_queue_a"), place_model("smoke_queue_b")
    folder = workspace.captures / f"smoke-queue-{os.getpid()}"
    c = create_app().test_client()
    try:
        check_capture_folder(c, str(a), folder)
        check_same_model_waits(c, str(a))
        check_models_overlap(c, str(a), str(b))
        check_bad_model(c)
        check_fill_size(c, str(a))
    finally:
        for p in (a, b):
            p.unlink(missing_ok=True)
        shutil.rmtree(folder, ignore_errors=True)
    print("OK")


def post(c, url, body):
    r = c.post(url, json=body)
    data = r.get_json()
    assert r.status_code == 200 and data.get("ok") is not False, (r.status_code, data)
    return data["data"]


def job(c, jid):
    return c.get(f"/api/jobs/{jid}").get_json()["data"]


def wait(c, jid, pred=lambda j: j["status"] in TERMINAL, what="the job to finish", timeout=120):
    t0 = time.time()
    while True:
        j = job(c, jid)
        if pred(j):
            return j
        assert time.time() - t0 < timeout, f"timed out waiting for {what}: {j['status']}"
        time.sleep(0.05)


def check_capture_folder(c, model, folder):
    """A queued run's images are saved by the client into one folder per queue."""
    done = wait(c, post(c, "/api/perform/sample", {"model_path": model, **FAST, "seed": 100,
                                                   "batch_size": 2})["job"]["id"])
    assert done["status"] == "done" and done["detail"]["seeds"] == [100, 101]
    for img, card in zip(done["detail"]["frames"], done["detail"]["cards"]):
        seed = card["params"]["seed"]
        out = post(c, "/api/perform/capture", {"image": img, "card": card, "folder": folder.name,
                                               "name": f"{done['id']}-seed{seed}"})
        assert Path(out["path"]).parent == folder, out["path"]
        back = read_params(Path(out["path"]))
        assert back["params"]["seed"] == seed and back["params"]["batch_size"] == 1, back
    listed = c.get("/api/captures").get_json()["data"]["captures"]
    mine = sorted(e["name"] for e in listed if e["folder"] == folder.name)
    assert mine == [f"{done['id']}-seed100", f"{done['id']}-seed101"], mine

    r = c.post("/api/perform/capture", json={"image": done["detail"]["frames"][0],
                                              "folder": "../outside"})
    assert r.status_code == 422, f"a folder outside captures was accepted ({r.status_code})"
    assert "group" not in done["detail"] and "saved" not in done["detail"]
    print("capture folder: one file per image with its recipe, listed under its folder")


def check_same_model_waits(c, model):
    """A second run on a busy model waits, and cancels without ever starting."""
    first = post(c, "/api/perform/sample", {"model_path": model, **SLOW, "seed": 1})["job"]
    second = post(c, "/api/perform/sample", {"model_path": model, **FAST, "seed": 2})["job"]
    assert first["status"] == "running", first["status"]
    assert second["status"] == "queued" and second["detail"]["queue"] == {"position": 1}

    wait(c, first["id"], lambda j: j["detail"].get("frame"), "a preview")
    light = c.get("/api/jobs?kinds=sample,inpaint&live=1&light=1").get_json()["data"]
    assert {j["id"] for j in light} == {first["id"], second["id"]}, [j["id"] for j in light]
    assert all(not any(k.startswith("frame") for k in j["detail"]) for j in light)
    full = c.get("/api/jobs?kind=sample").get_json()["data"]
    assert any("frame" in j["detail"] for j in full), "the full list lost its frames"

    assert c.post(f"/api/jobs/{second['id']}/cancel").get_json()["data"]["cancelled"]
    assert job(c, second["id"])["status"] == "cancelled"
    assert c.post(f"/api/jobs/{first['id']}/cancel").get_json()["data"]["cancelled"]
    assert wait(c, first["id"])["status"] == "cancelled"
    time.sleep(0.2)
    j = job(c, second["id"])
    assert j["status"] == "cancelled" and j["progress"] == 0, "a cancelled queued run ran"
    print("same model: the second run queues, the light list drops images, cancel never starts it")


def check_models_overlap(c, a, b):
    """Runs on two different models are both running at once."""
    ja = post(c, "/api/perform/sample", {"model_path": a, **SLOW, "seed": 1})["job"]
    jb = post(c, "/api/perform/sample", {"model_path": b, **SLOW, "seed": 1})["job"]
    assert (ja["status"], jb["status"]) == ("running", "running"), (ja["status"], jb["status"])
    for j in (ja, jb):
        c.post(f"/api/jobs/{j['id']}/cancel")
    for j in (ja, jb):
        wait(c, j["id"])
    print("different models: two lanes run at once")


def check_bad_model(c):
    """A model that cannot be read is now a failed job, not a failed request."""
    out = post(c, "/api/perform/sample", {"model_path": str(workspace.models / "nope.pt"), **FAST})
    j = wait(c, out["job"]["id"])
    assert j["status"] == "error" and "not found" in j["message"], (j["status"], j["message"])
    print("bad model: the job fails with the reason")


def check_fill_size(c, model):
    """A fill's size is worked out in its lane and still reaches the job and card."""
    from PIL import Image

    from utils.imaging import data_url

    canvas = data_url(Image.new("RGB", (72, 72), (90, 90, 90)))
    mask = data_url(Image.new("L", (72, 72), 255))
    out = post(c, "/api/perform/inpaint", {"model_path": model, **FAST, "init_image": canvas,
                                           "mask": mask})
    j = wait(c, out["job"]["id"])
    assert j["status"] == "done", j["message"]
    assert j["detail"]["fill_size"] == j["detail"]["card"]["fill_size"], j["detail"]["card"]
    assert j["detail"]["canvas_size"] == [72, 72]
    print("fill: fill_size computed in the lane, on the job and the card")


if __name__ == "__main__":
    main()
