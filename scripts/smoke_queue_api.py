"""API smoke for generations going through the lanes (CPU, untrained models).

Covers what the routes add on top of scripts/smoke_lanes.py: jobs start
queued, a repeat run's seeds and saved files, cancelling a queued job and a
whole group, the light job list, and a bad model failing as a job.
"""
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    group_dir = workspace.captures / f"smoke-repeat-{os.getpid()}"
    c = create_app().test_client()
    try:
        check_repeat(c, str(a), group_dir)
        check_same_model_waits(c, str(a))
        check_models_overlap(c, str(a), str(b))
        check_cancel_group(c, str(a))
        check_bad_model(c)
        check_fill_size(c, str(a))
    finally:
        for p in (a, b):
            p.unlink(missing_ok=True)
        shutil.rmtree(group_dir, ignore_errors=True)
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


def check_repeat(c, model, group_dir):
    """Runs k start at seed + k * variations, and each image is saved with its recipe."""
    out = post(c, "/api/perform/sample", {
        "model_path": model, **FAST, "seed": 100, "batch_size": 2, "repeat": 3,
        "group_name": group_dir.name})
    jobs = out["jobs"]
    assert len(jobs) == 3 and out["job"]["id"] == jobs[0]["id"]
    groups = [j["detail"]["group"] for j in jobs]
    assert len({g["id"] for g in groups}) == 1 and [g["index"] for g in groups] == [0, 1, 2]
    assert [j["detail"]["seeds"] for j in jobs] == [[100, 101], [102, 103], [104, 105]]
    # One model, so one lane: the rest wait their turn.
    assert [j["status"] for j in jobs[1:]] == ["queued", "queued"], [j["status"] for j in jobs]

    done = [wait(c, j["id"]) for j in jobs]
    assert all(j["status"] == "done" for j in done), [j["message"] for j in done]
    for j in done:
        names = sorted(Path(p).name for p in j["detail"]["saved"])
        assert names == sorted(f"{j['id']}-seed{s}.png" for s in j["detail"]["seeds"]), names
        for p, s in zip(sorted(j["detail"]["saved"]), sorted(j["detail"]["seeds"])):
            card = read_params(Path(p))
            assert card["params"]["seed"] == s and card["params"]["batch_size"] == 1, card
    assert len(list(group_dir.glob("*.png"))) == 6

    listed = c.get("/api/captures").get_json()["data"]["captures"]
    mine = [e for e in listed if e["folder"] == group_dir.name]
    assert len(mine) == 6, f"/captures lists {len(mine)} of the run's 6 images"

    single = post(c, "/api/perform/sample", {"model_path": model, **FAST, "seed": 7})
    assert "jobs" in single and "group" not in single["job"]["detail"]
    assert "saved" not in wait(c, single["job"]["id"])["detail"], "a single run was saved"
    print("repeat: consecutive seeds, one file per image with its recipe, /captures sees them")


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


def check_cancel_group(c, model):
    """Cancelling a group stops the running run and every waiting one."""
    out = post(c, "/api/perform/sample", {"model_path": model, **SLOW, "seed": 1, "repeat": 4,
                                          "group_name": f"smoke-cancel-{os.getpid()}"})
    gid = out["job"]["detail"]["group"]["id"]
    wait(c, out["job"]["id"], lambda j: j["status"] == "running", "the first run to start")
    r = c.post(f"/api/jobs/group/{gid}/cancel").get_json()["data"]
    assert r["cancelled"] == 4, r
    finals = [wait(c, j["id"]) for j in out["jobs"]]
    assert all(j["status"] == "cancelled" for j in finals), [j["status"] for j in finals]
    assert not any(j["detail"].get("saved") for j in finals)
    assert not (workspace.captures / f"smoke-cancel-{os.getpid()}").exists()
    print("group cancel: the running run and all three waiting ones stop, nothing saved")


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
