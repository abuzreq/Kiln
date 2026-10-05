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

for n in ("mergeA", "mergeB", "mergeC", "merged_linear", "merged_slerp", "merged_blockwise",
          "confA", "confB", "confC", "merged_conf", "twoA", "twoB", "ends", "which"):
    (workspace.models / f"{n}.pt").unlink(missing_ok=True)
(workspace.recipes / "merged_blockwise.json").unlink(missing_ok=True)
print("OK")
