"""Smoke test 2-way merge via the API using two synthetic models."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

for n in ("mergeA", "mergeB", "mergeC", "merged_linear", "merged_slerp", "merged_blockwise",
          "confA", "confB", "confC", "merged_conf"):
    (workspace.models / f"{n}.pt").unlink(missing_ok=True)
(workspace.recipes / "merged_blockwise.json").unlink(missing_ok=True)
print("OK")
