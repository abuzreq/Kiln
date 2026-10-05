"""How related two models are, measured from their weights alone.

The figures are taken on the slot the sampler reads (EMA where there is one) and
on the network's own tensors (``backend.net_state``), per UNet stage and overall:

cosine      of the two weight vectors: near 1 for one model slightly moved,
            well above 0 for models with a common ancestor, near 0 for
            models trained apart -- their units are in unrelated orders.
distance    ||A - B|| / ||A||.

Given a base, the same for the task vectors tau = model - base, which is what
TIES and its relatives work on:

drift       ||tau|| / ||base|| for each model: how far it moved from the base.
cosine      of tau_A and tau_B.
agreement   among the entries both models' TIES trim keeps (the top 20% of
            each tau by magnitude, per tensor), the share whose signs agree.
overlap     how many entries the two trims keep in common, of all they keep.

Nothing here needs data or a GPU; two small models take well under a second.
"""
import math
import threading
from collections import OrderedDict
from pathlib import Path

from app.core import backends, library

STAGES = ("encoder", "mid", "decoder", "other")
#: TIES' default trim: the share of each task vector it keeps.
TOP = 0.2


def _slot(path: str, ema: bool):
    backend, ref = backends.resolve(path)
    slots = backend.merge_slots(ref)
    return backend, backend.net_state(slots[backend.sampled_slot(slots, ema)])


def _new():
    return {"ab": 0.0, "aa": 0.0, "bb": 0.0, "dd": 0.0, "n": 0,
            "tt": 0.0, "ta": 0.0, "tb": 0.0, "cc": 0.0, "both": 0, "agree": 0,
            "kept_a": 0, "kept_b": 0}


def _top_mask(t):
    """The entries a TIES trim keeps: the top ``TOP`` share by magnitude, never zeros."""
    mag = t.abs()
    k = max(1, round(t.numel() * TOP))
    thr = mag.kthvalue(t.numel() - k + 1).values  # the k-th largest; faster than topk
    return (mag >= thr) & (mag > 0)


def _ratio(x, y):
    return round(x / y, 4) if y > 0 else None


def _cos(dot, aa, bb):
    return round(dot / math.sqrt(aa * bb), 4) if aa > 0 and bb > 0 else None


def _figures(acc: dict, with_base: bool) -> dict:
    out = {
        "cosine": _cos(acc["ab"], acc["aa"], acc["bb"]),
        "distance": _ratio(math.sqrt(acc["dd"]), math.sqrt(acc["aa"])),
        "params": acc["n"],
    }
    if with_base:
        kept = acc["kept_a"] + acc["kept_b"] - acc["both"]
        out["base"] = {
            "drift_a": _ratio(math.sqrt(acc["ta"]), math.sqrt(acc["cc"])),
            "drift_b": _ratio(math.sqrt(acc["tb"]), math.sqrt(acc["cc"])),
            "cosine": _cos(acc["tt"], acc["ta"], acc["tb"]),
            "agreement": _ratio(acc["agree"], acc["both"]),
            "overlap": _ratio(acc["both"], kept),
        }
    return out


def _measure(path_a: str, path_b: str, base: str | None, ema: bool) -> dict:
    backend, sa = _slot(path_a, ema)
    _, sb = _slot(path_b, ema)
    sc = _slot(base, ema)[1] if base else None
    return {**measure(backend, sa, sb, sc), "base": base}


def measure(backend, sa: dict, sb: dict, sc: dict | None = None) -> dict:
    """The figures for two network states already in memory (and a base's)."""
    import torch

    acc = {s: _new() for s in (*STAGES, "overall")}
    for k, ta in sa.items():
        tb = sb.get(k)
        tc = sc.get(k) if sc is not None else None
        if not (torch.is_tensor(ta) and torch.is_tensor(tb)) or ta.shape != tb.shape \
                or not ta.is_floating_point():
            continue
        a, b = ta.double().flatten(), tb.double().flatten()
        rows = [acc[backend.stage_of_key(k)], acc["overall"]]
        figs = {"ab": float(a @ b), "aa": float(a @ a), "bb": float(b @ b),
                "dd": float((a - b) @ (a - b)), "n": a.numel()}
        if sc is not None and torch.is_tensor(tc) and tc.shape == ta.shape:
            c = tc.double().flatten()
            da, db = a - c, b - c
            ka, kb = _top_mask(da), _top_mask(db)
            both = ka & kb
            figs.update(tt=float(da @ db), ta=float(da @ da), tb=float(db @ db),
                        cc=float(c @ c), both=int(both.sum()),
                        agree=int((both & (torch.sign(da) == torch.sign(db))).sum()),
                        kept_a=int(ka.sum()), kept_b=int(kb.sum()))
        for row in rows:
            for f, v in figs.items():
                row[f] += v
    out = {s: _figures(acc[s], sc is not None) for s in STAGES if acc[s]["n"]}
    return {"overall": _figures(acc["overall"], sc is not None), "stages": out}


_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_CACHE_MAX = 16
_lock = threading.Lock()


def _stamp(path: str | None):
    if not path:
        return None
    f = library.weights_file(path)
    return (str(Path(path).resolve()), *(library.file_stamp(f) if f else ()))


def stats(path_a: str, path_b: str, base: str | None = None, ema: bool = True) -> dict:
    """The figures above for A and B, and for their task vectors when a base is given.

    Kept per file state, so the tab can ask on every pair change: a file that
    changes on disk is measured again.
    """
    key = (_stamp(path_a), _stamp(path_b), _stamp(base), bool(ema))
    with _lock:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    res = _measure(path_a, path_b, base, ema)
    with _lock:
        _CACHE[key] = res
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return res


# --- a base from lineage -----------------------------------------------------
# Roles that make a parent an ancestor of the same weights. A merge's A and B
# are two ancestors at once and a base is not chosen through them.
_ANCESTRAL = ("start", "base", "source")


def _recorded_lineage(path: str) -> tuple[dict, str]:
    """A model's lineage, and how it is known: from its card, or best effort.

    Models saved before cards recorded lineage still name the run they came
    from (``origin_run``); that run's run.json says what it started from.
    """
    lin = library.lineage_of(path)
    if lin.get("kind") != "unknown":
        return lin, "lineage"
    run = library.read_card(path).get("origin_run")
    if run:
        from app.backend.data import projects

        try:
            found = library.run_lineage(projects.find_run(run))
        except Exception:  # noqa: BLE001 -- the run may be gone
            found = None
        if found:
            return found, "run.json"
    return lin, "lineage"


def _ancestry(path: str, depth: int = 8) -> list[dict]:
    """``path`` and then its ancestors, nearest first, as far as they resolve."""
    chain = [{"path": str(path), "key": library.fingerprint(path) or str(path),
              "how": "self"}]
    seen = {chain[0]["key"]}
    cur = str(path)
    for _ in range(depth):
        lin, how = _recorded_lineage(cur)
        parent = next((p for p in lin.get("parents") or [] if p.get("role") in _ANCESTRAL), None)
        if parent is None:
            break
        where = library.resolve_parent(parent)
        if where is None:
            break
        key = parent.get("fingerprint") or library.fingerprint(where) or where
        if key in seen:
            break
        seen.add(key)
        chain.append({"path": where, "key": key, "how": how, "name": parent.get("name")})
        cur = where
    return chain


def suggest_base(path_a: str, path_b: str) -> dict | None:
    """The nearest model A and B both descend from, if their records say so.

    When one is the other's ancestor, that one is the base: the other's task
    vector is then everything it learned since. None when nothing links them.
    """
    from app.core.craft.merging import check_compat

    chain_a, chain_b = _ancestry(path_a), _ancestry(path_b)
    if chain_a[0]["key"] == chain_b[0]["key"]:
        return None  # the same weights twice
    where_b = {n["key"]: i for i, n in enumerate(chain_b)}
    best = None
    for i, node in enumerate(chain_a):
        j = where_b.get(node["key"])
        if j is not None and (best is None or i + j < best[0]):
            best = (i + j, i, j, node)
    if best is None:
        return None
    _, i, j, node = best
    if not check_compat(path_a, node["path"])["compatible"]:
        return None
    relation = "a_is_ancestor" if i == 0 else "b_is_ancestor" if j == 0 else "shared"
    how = node["how"] if node["how"] != "self" else chain_b[j]["how"]
    return {"path": node["path"], "name": node.get("name") or Path(node["path"]).stem,
            "relation": relation, "how": how}
