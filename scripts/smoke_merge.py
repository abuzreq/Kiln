"""Smoke test 2-way merge via the API using two synthetic models."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import _scratch_workspace  # noqa: E402,F401  -- before app: never write to ~/kiln

import torch
from app.core.engine.arch import build_unet
from app.core.config import workspace
from app.backend.app import create_app


def make(name, seed):
    torch.manual_seed(seed)
    unet = build_unet("tinyunet_with_attention3", [1, 2, 2, 2])
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    p = workspace.models / f"{name}.pt"
    torch.save({"step": 0, "model": state, "ema": state, "mults": [1, 2, 2, 2],
                "mtype": "tinyunet_with_attention3", "pred": "x0"}, p)
    return str(p)


a = make("mergeA", 1)
b = make("mergeB", 2)
c = create_app().test_client()

chk = c.post("/api/craft/merge/check", json={"model_a": a, "model_b": b}).get_json()["data"]
print("compatible:", chk["compatible"])

for method in ("linear", "slerp", "blockwise"):
    r = c.post("/api/craft/merge", json={
        "model_a": a, "model_b": b, "out_name": f"merged_{method}",
        "method": method, "alpha": 0.5,
        "block_weights": {"encoder": 0.2, "mid": 0.5, "decoder": 0.8},
        "save_recipe": method == "blockwise",
    }).get_json()
    print(method, "->", r["ok"], Path(r["data"]["path"]).name)

recipes = c.get("/api/library/recipes").get_json()["data"]
print("recipes:", [x["name"] for x in recipes])

# incompatibility guardrail
d = make("mergeC", 3)
import torch as _t
ck = _t.load(d, map_location="cpu", weights_only=False)
ck["mults"] = [1, 2, 4, 4]
_t.save(ck, d)
bad = c.post("/api/craft/merge/check", json={"model_a": a, "model_b": d}).get_json()["data"]
print("guardrail incompatible:", not bad["compatible"], bad["reasons"])

# configurable attention: the layout is part of the shape, so two conf models
# merge only when it matches, and the result carries it
import argparse

from app.core.backends.xurdif import attn as attn_spec


def make_conf(name, seed, spec):
    torch.manual_seed(seed)
    unet = build_unet("tinyunet_conf_attention", [1, 2, 2, 2], attn_config=attn_spec.parse(spec))
    state = {f"denoise_fn.{k}": v for k, v in unet.state_dict().items()}
    p = workspace.models / f"{name}.pt"
    # recorded the way upstream's trainer records it: attn_conf plus opt
    torch.save({"step": 0, "model": state, "ema": state, "mults": [1, 2, 2, 2],
                "mtype": "tinyunet_conf_attention", "pred": "x0", "attn_conf": spec,
                "opt": argparse.Namespace(model="tinyunet_conf_attention", mults=[1, 2, 2, 2],
                                          pred="x0", attn=spec, attn_config=attn_spec.parse(spec))}, p)
    return str(p)


ca = make_conf("confA", 4, "-1:linear,mid:full")
cb = make_conf("confB", 5, "mid:full,-1:linear")     # same layout, spelled differently
cc = make_conf("confC", 6, "mid:full")
same = c.post("/api/craft/merge/check", json={"model_a": ca, "model_b": cb}).get_json()["data"]
assert same["compatible"], same["reasons"]
diff = c.post("/api/craft/merge/check", json={"model_a": ca, "model_b": cc}).get_json()["data"]
assert not diff["compatible"] and any("attention layouts differ" in r for r in diff["reasons"]), diff["reasons"]
r = c.post("/api/craft/merge", json={"model_a": ca, "model_b": cb, "out_name": "merged_conf",
                                     "method": "linear", "alpha": 0.5}).get_json()
assert r["ok"], r
merged = _t.load(r["data"]["path"], map_location="cpu", weights_only=False)
assert merged["attn_conf"] == "-1:linear,mid:full", merged.keys()
assert merged["opt"].attn_config == {-1: "linear", "mid": "full"}, merged["opt"]
assert merged["opt"].model == "tinyunet_conf_attention" and merged["opt"].mults == [1, 2, 2, 2]
print("conf guardrail:", diff["reasons"], "| merged conf carries", merged["attn_conf"])

# Exact ends, and "Weights to merge" honoured. These fixtures carry distinct
# model and ema weights; the ones above share one state, which is how "EMA
# only" blending both slots went unnoticed.
from app.core.craft.merging import merge as do_merge


def make_two_slot(name, seed):
    torch.manual_seed(seed)
    raw = build_unet("tinyunet_with_attention3", [1, 2, 2, 2]).state_dict()
    ema = build_unet("tinyunet_with_attention3", [1, 2, 2, 2]).state_dict()
    p = workspace.models / f"{name}.pt"
    torch.save({"step": 0, "mults": [1, 2, 2, 2], "mtype": "tinyunet_with_attention3",
                "pred": "x0",
                "model": {f"denoise_fn.{k}": v for k, v in raw.items()},
                "ema": {f"denoise_fn.{k}": v for k, v in ema.items()}}, p)
    return str(p)


def slots_of(path):
    data = _t.load(path, map_location="cpu", weights_only=False)
    return {k: data[k] for k in ("model", "ema")}


def same(x, y):
    return all(torch.equal(x[k], y[k]) for k in x)


ta, tb = make_two_slot("twoA", 7), make_two_slot("twoB", 8)
sa, sb = slots_of(ta), slots_of(tb)
assert not same(sa["model"], sa["ema"]), "fixture slots must differ"
all0 = {"encoder": 0.0, "mid": 0.0, "decoder": 0.0}
all1 = {"encoder": 1.0, "mid": 1.0, "decoder": 1.0}
for method, alpha, blocks, want, label in (
    ("linear", 0.0, {}, sa, "linear a=0 -> A"),
    ("linear", 1.0, {}, sb, "linear a=1 -> B"),
    ("slerp", 0.0, {}, sa, "slerp a=0 -> A"),
    ("slerp", 1.0, {}, sb, "slerp a=1 -> B"),
    ("blockwise", 0.7, all0, sa, "blockwise all-0 -> A"),
    ("blockwise", 0.3, all1, sb, "blockwise all-1 -> B"),
):
    out = slots_of(do_merge(ta, tb, "ends", method=method, alpha=alpha,
                            block_weights=blocks)["path"])
    for slot in ("model", "ema"):
        assert same(out[slot], want[slot]), f"{label}: {slot} differs"
    print(f"  {label:24s} exact in both slots")

for which, kept in (("ema", "model"), ("model", "ema")):
    out = slots_of(do_merge(ta, tb, "which", method="linear", alpha=0.5, which=which)["path"])
    assert same(out[kept], sa[kept]), f"which={which}: {kept} should be A's"
    assert not same(out[which], sa[which]), f"which={which}: {which} was not blended"
    want = {k: torch.lerp(sa[which][k].float(), sb[which][k].float(), 0.5) for k in sa[which]}
    assert same(out[which], want), f"which={which}: {which} is not the linear blend"
    print(f"  which={which:5s} blends {which}, keeps A's {kept}")
bad = c.post("/api/craft/merge", json={"model_a": ta, "model_b": tb, "out_name": "x",
                                       "which": "neither"}).get_json()
assert not bad["ok"], bad
print("weights to merge: honoured, and an unknown choice is refused")


def check_ladder_matches_saved(device):
    """A rung blended in memory samples exactly like the merge saved to disk."""
    import time

    from app.core.craft.ladder import LadderRig
    from app.core.engine.sampler import SampleParams, sampler
    from app.core.model_manager import manager

    def x0(path, bundle=None):
        p = SampleParams(model_path=path, image_size=32, steps=6, seed=1234, device=device)
        last = None
        for last in sampler.run(p, bundle=bundle):
            pass
        return last._x.detach().cpu()

    cached = manager.load(ta, device=device, ema=True)["model"]
    before = {k: v.clone() for k, v in cached.state_dict().items()}
    plain_a = x0(ta)
    rig = LadderRig(ta, tb, device)
    blocks = {"encoder": 0.2, "mid": 0.6, "decoder": 0.9}
    for method in ("linear", "slerp", "blockwise"):
        saved = do_merge(ta, tb, f"rung_{method}", method=method, alpha=0.3,
                         block_weights=blocks)["path"]
        want = x0(saved)
        manager.evict(saved)
        sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
        sync()
        t0 = time.perf_counter()
        bundle = rig.apply(method, 0.3, blocks)
        sync()
        t1 = time.perf_counter()
        got = x0(ta, bundle)
        sync()
        t2 = time.perf_counter()
        assert torch.equal(got, want), f"{device} {method}: rung differs from the saved merge"
        assert not torch.equal(got, plain_a), f"{device} {method}: rung sampled plain A"
        print(f"  {device:4s} {method:9s} rung == saved merge, bit for bit "
              f"(blend+load {1000 * (t1 - t0):.0f} ms, sample {1000 * (t2 - t1):.0f} ms)")
        (workspace.models / f"rung_{method}.pt").unlink(missing_ok=True)
    after = cached.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "the cached A was changed"
    print(f"  {device:4s} the manager's cached model A is untouched")


check_ladder_matches_saved("cpu")
if torch.cuda.is_available():
    check_ladder_matches_saved("cuda")
print("ladder rungs: identical to saved merges")


def check_ladder_plan():
    from app.core.craft.ladder import plan
    from utils.exceptions import ValidationError

    five = [0, 0.25, 0.5, 0.75, 1]
    cells = plan("linear", {}, [{"param": "alpha", "values": five}])
    assert [c["same_as"] for c in cells] == ["a", None, None, None, "b"]
    assert [c["order"] for c in cells] == [None, 1, 0, 2, None], "not middle-out"
    two = plan("slerp", {}, [{"param": "method", "values": ["slerp", "linear"]},
                             {"param": "alpha", "values": five}])
    assert len(two) == 10 and sum(c["order"] is not None for c in two) == 6
    assert [c["recipe"]["method"] for c in two if c["order"] in (0, 1)] == ["slerp", "linear"]
    dup = plan("linear", {}, [{"param": "alpha", "values": [0.5, 0.3, 0.5]}])
    assert dup[2]["same_as"] == 0 and dup[2]["order"] is None, dup
    zoom = plan("linear", {}, [{"param": "alpha", "values": [0.25, 0.3125, 0.375, 0.4375, 0.5]}],
                known=[{"method": "linear", "alpha": 0.25}, {"method": "linear", "alpha": 0.5}])
    assert [c["same_as"] for c in zoom] == ["known", None, None, None, "known"], zoom
    refine = plan("blockwise", {"block_weights": {"mid": 0.5}},
                  [{"param": "encoder", "values": [0, 0.25, 0.5]},
                   {"param": "decoder", "values": [0.5, 0.75, 1]}],
                  known=[{"method": "blockwise",
                          "block_weights": {"encoder": 0.25, "mid": 0.5, "decoder": 0.75}}])
    assert refine[4]["same_as"] == "known" and sum(c["order"] is not None for c in refine) == 8
    grid = plan("linear", {"block_weights": {"mid": 0.5}},
                [{"param": "encoder", "values": [0, 0.5, 1]},
                 {"param": "decoder", "values": [0, 0.5, 1]}])
    assert all(c["recipe"]["method"] == "blockwise" for c in grid)
    assert all(c["same_as"] is None for c in grid), "mid held at 0.5: no corner is A or B"
    assert grid[4]["order"] == 0, "the centre of a grid comes first"
    corners = plan("blockwise", {"block_weights": {"mid": 0}},
                   [{"param": "encoder", "values": [0, 1]}, {"param": "decoder", "values": [0, 1]}])
    assert corners[0]["same_as"] == "a" and corners[3]["same_as"] is None
    for bad in (
        [{"param": "alpha", "values": [0.1] * 26}],
        [{"param": "alpha", "values": [0.1] * 6}, {"param": "method", "values": ["a"] * 5}],
        [{"param": "encoder", "values": [0]}, {"param": "method", "values": ["linear"]}],
        [{"param": "alpha", "values": [1.5]}],
        [{"param": "zoom", "values": [0]}],
    ):
        try:
            plan("linear", {}, bad)
        except ValidationError:
            continue
        raise AssertionError(f"plan accepted {bad}")
    print("ladder plan: ends reuse A and B, known and duplicate cells collapse, middle first, "
          "bad axes refused")


def check_ladder_route():
    import time

    def wait(job_id, until=lambda j: j["status"] in ("done", "error", "cancelled"), limit=300):
        t0 = time.time()
        while True:
            j = c.get(f"/api/jobs/{job_id}").get_json()["data"]
            if until(j):
                return j
            assert time.time() - t0 < limit, f"job stuck: {j['status']} {j['message']}"
            time.sleep(0.1)

    sample = {"image_size": 32, "steps": 4, "seed": None}
    five = [0, 0.25, 0.5, 0.75, 1]
    r = c.post("/api/craft/merge/ladder", json={
        "model_a": ta, "model_b": tb, "method": "slerp",
        "axes": [{"param": "method", "values": ["slerp", "linear"]},
                 {"param": "alpha", "values": five}],
        "sample": sample,
    }).get_json()
    assert r["ok"], r
    # Poll as the Merge tab does: each picture once, by revision.
    jid, since, got, sent = r["data"]["job"]["id"], None, {}, 0
    while True:
        q = f"/api/jobs/{jid}" + ("" if since is None else f"?since={since}")
        j = c.get(q).get_json()["data"]
        since = j["detail"]["rev"]
        entries = [(f"ref {k}", v) for k, v in (j["detail"]["refs"] or {}).items() if v]
        entries += [(f"cell {n}", x) for n, x in enumerate(j["detail"]["cells"])]
        for key, x in entries:
            if x.get("image"):
                assert key not in got, f"{key} sent twice"
                got[key] = x["image"]
                sent += 1
        if j["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)
    j = c.get(f"/api/jobs/{jid}").get_json()["data"]
    assert j["status"] == "done", j["message"]
    d = j["detail"]
    full = {f"ref {k}": v["image"] for k, v in d["refs"].items()}
    full |= {f"cell {n}": x["image"] for n, x in enumerate(d["cells"]) if x["image"]}
    assert got == full and sent == d["rev"] == 8, (sorted(got), sorted(full), sent, d["rev"])
    print("ladder polls: each of 8 pictures sent once by revision, same set as a full poll")
    assert (d["rows"], d["cols"]) == (2, 5) and len(d["cells"]) == 10
    seeds = {d["refs"][k]["card"]["params"]["seed"] for k in "ab"}
    seeds |= {x["card"]["params"]["seed"] for x in d["cells"] if x["card"]}
    assert seeds == {d["seed"]}, f"a blank seed must resolve once, got {seeds}"
    sampled = [x for x in d["cells"] if x["image"]]
    assert len(sampled) == 6 and all(x["same_as"] is None for x in sampled)
    assert {x["same_as"] for x in d["cells"] if not x["image"]} == {"a", "b"}
    card = sampled[0]["card"]
    picked = sampled[0]
    assert card["kind"] == "merge-ladder" and "model_path" not in card
    assert card["merge"]["model_a"] == ta and card["merge"]["method"] in ("slerp", "linear")
    print(f"ladder job: 2 refs + 6 merges on seed {d['seed']}, ends shared, cards carry the recipe")

    r = c.post("/api/craft/merge/ladder", json={
        "model_a": ta, "model_b": tb, "refs": False,
        "axes": [{"param": "alpha", "values": [0.3, 0.4]}], "sample": {**sample, "seed": 7},
    }).get_json()
    j = wait(r["data"]["job"]["id"])
    assert j["detail"]["refs"] is None and j["detail"]["seed"] == 7
    assert all(x["image"] for x in j["detail"]["cells"])
    print("ladder job: a zoom without refs samples only its own steps, on the seed it was given")

    bad = c.post("/api/craft/merge/ladder", json={"model_a": a, "model_b": d_bad,
                 "axes": [{"param": "alpha", "values": [0.5]}]})
    assert bad.status_code == 409 and not bad.get_json()["ok"], bad.get_json()
    big = c.post("/api/craft/merge/ladder", json={"model_a": ta, "model_b": tb,
                 "axes": [{"param": "alpha", "values": [0.5] * 30}]})
    assert big.status_code == 422 and "25" in big.get_json()["error"], big.get_json()
    print("ladder route: incompatible pair refused (409), oversized ladder refused (422)")

    r = c.post("/api/craft/merge/ladder", json={
        "model_a": ta, "model_b": tb, "refs": False,
        "axes": [{"param": "alpha", "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}],
        "sample": {**sample, "steps": 30},
    }).get_json()
    jid = r["data"]["job"]["id"]
    wait(jid, until=lambda j: any(x["image"] for x in j["detail"]["cells"])
         or j["status"] != "running" and j["status"] != "queued")
    c.post(f"/api/jobs/{jid}/cancel")
    j = wait(jid)
    kept = sum(bool(x["image"]) for x in j["detail"]["cells"])
    assert j["status"] == "cancelled" and 1 <= kept < 9, (j["status"], kept)
    print(f"ladder cancel: stopped with {kept} of 9 merges kept")
    return picked


def check_save_from_ladder(cell):
    """A saved pick keeps its sample as the thumbnail, and that recipe replays."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    from app.core import library
    from app.core.engine.sampler import SampleParams, sampler
    from utils.imaging import read_params

    def pixels(im):
        if isinstance(im, str):
            im = Image.open(io.BytesIO(base64.b64decode(im.split(",", 1)[1])))
        return np.asarray(im.convert("RGB"))

    r = cell["recipe"]
    save = {"model_a": ta, "model_b": tb, "method": r["method"], "alpha": r["alpha"],
            "thumbnail": cell["image"], "card": cell["card"]}
    res = c.post("/api/craft/merge", json={**save, "out_name": "picked"}).get_json()["data"]
    assert res.get("thumbnail") and "thumbnail_skipped" not in res, res
    back = read_params(library.thumb_path(res["path"]))
    assert back["model_path"] == res["path"] and back["model"] == "picked", back
    assert back["merge"]["method"] == r["method"], back
    p = back["params"]
    last = None
    for last in sampler.run(SampleParams(model_path=back["model_path"], image_size=p["image_size"],
                                         steps=p["steps"], seed=p["seed"], eta=p["eta"],
                                         sampler=p["sampler"], ema=p["ema"])):
        pass
    assert (pixels(last["image_pp"]) == pixels(cell["image"])).all(), "thumbnail does not replay"
    print("save: the thumbnail's recipe names the saved model and replays to the same pixels")

    res = c.post("/api/craft/merge", json={**save, "out_name": "rawonly",
                                           "which": "model"}).get_json()["data"]
    assert "thumbnail" not in res and "ema" in res["thumbnail_skipped"], res
    assert not library.thumb_path(res["path"]).exists()
    print("save: raw-only weights skip an EMA sample as thumbnail:", res["thumbnail_skipped"])


d_bad = d
check_ladder_plan()
check_save_from_ladder(check_ladder_route())

for n in ("mergeA", "mergeB", "mergeC", "merged_linear", "merged_slerp", "merged_blockwise",
          "confA", "confB", "confC", "merged_conf", "twoA", "twoB", "ends", "which",
          "picked", "rawonly"):
    (workspace.models / f"{n}.pt").unlink(missing_ok=True)
(workspace.recipes / "merged_blockwise.json").unlink(missing_ok=True)
print("OK")
