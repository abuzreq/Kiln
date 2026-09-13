"""Novelty explorer: random bend stacks on one model, kept when they look new.

A background worker, shaped like ``app.core.previews``: one daemon thread, an
in-memory run state, an on-disk archive per model. It samples random bend
stacks at a small fixed recipe, embeds the renders with the CLIP that guidance
already loads, and keeps a candidate when it sits far from everything kept so
far (novelty search, Lehman & Stanley, with a threshold that adapts to how
often candidates get in).

Two rules worth knowing before changing anything here:

- There is no GPU admission control, by design. Sampling beside training is a
  supported workflow and the CUDA-OOM fallback is the pressure valve. What this
  module does have is a *courtesy yield*: between candidates it sleeps while an
  interactive sample or inpaint job is running, so the user's own run has the
  GPU to itself. That is a politeness rule about latency, nothing more.
- The archive lives under ``workspace.cache/discoveries/<key>`` with the same
  key previews use (path + mtime + size), so a retrained model gets a fresh
  archive and the old one stays browsable under its old model name. Every entry
  records the model that made it, as does every PNG's card, so a discovery
  dragged onto the canvas lands on that model.
"""
import json
import os
import random
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.core import library, previews
from app.core.config import workspace
from app.core.craft import bending
from app.core.craft import ops as ops_mod
from app.core.engine.sampler import SampleParams, _Clip, pick_device, sampler
from app.core.model_manager import manager
from utils.imaging import build_card, save_with_params
from utils.logger import get_logger
from utils.process_control import registry

log = get_logger("explore")

# The evaluation recipe matches previews so pictures are comparable across
# models. One denoise loop yields SEEDS seeds (42, 43, 44); the embedding is
# their average, so novelty is about the bend, not one seed's content.
FIRST_SEED = 42
SEEDS = 3
SIZE = 256
STEPS = 16
SAMPLER = "unipc"

K = 5                      # nearest neighbours the novelty score averages over
CAP = 100                  # entries per archive; past it the oldest non-starred one goes
WINDOW = 20                # tries per threshold adaptation
WINDOW_HIGH = 4            # more accepts than this in a window: raise the bar
INJECT_P = 0.05            # chance to keep a merely-different candidate anyway
MAX_BENDS = 3

# What "looks different" is measured with. Embeddings from different metrics
# are not comparable, so each metric keeps its own archive per model. The
# distance scales differ too: DINOv2 spreads images further apart than CLIP
# does, so its floors start higher and the adaptive threshold does the rest.
#   baseline_eps  below this from the unbent render, a bend did nothing at all
#   min_distance  below this, it did nothing worth keeping
#   threshold0    where the adaptive threshold starts for a new archive
#   threshold     the range the adaptive threshold may move in
# The DINOv2 numbers come from a first run on a real model: mean distances to
# the five nearest kept entries sat between 0.43 and 0.72 where CLIP's sit
# between 0.08 and 0.3, so the same ceiling would have let everything in.
METRICS = {
    "clip": {"label": "CLIP", "baseline_eps": 0.01, "min_distance": 0.03,
             "threshold0": 0.08, "threshold": (0.02, 0.4)},
    "dinov2": {"label": "DINOv2", "baseline_eps": 0.05, "min_distance": 0.15,
               "threshold0": 0.45, "threshold": (0.1, 0.9)},
}
DEFAULT_METRIC = "clip"

YIELD_KINDS = ("sample", "inpaint", "randomize", "bend_sweep", "sweep")
GROUP_WEIGHTS = {"encoder": 0.25, "mid": 0.25, "decoder": 0.25,
                 "attention": 0.1, "blocks": 0.05, "all": 0.05}
STACK_SIZES = ((1, 0.5), (2, 0.35), (3, 0.15))


# --- archive -----------------------------------------------------------
def archive_root() -> Path:
    return workspace.cache / "discoveries"


def archive_dir(model_path: str | Path, metric: str = DEFAULT_METRIC) -> Path | None:
    """One folder per model and metric. CLIP keeps the bare key, so archives
    made before the metric toggle existed are still found."""
    key = previews.cache_key(model_path)
    if not key:
        return None
    return archive_root() / (key if metric == DEFAULT_METRIC else f"{key}-{metric}")


def model_exists(model_path: str) -> bool:
    return bool(model_path) and (Path(model_path).exists() or ":" in str(model_path))


def model_name(model_path: str) -> str:
    try:
        name = (library.read_card(model_path) or {}).get("name")
        if name:
            return name
    except Exception:  # noqa: BLE001
        pass
    return Path(str(model_path)).stem


def _empty_index(model_path: str, metric: str) -> dict:
    return {
        "version": 1, "model_path": str(model_path), "model": {}, "metric": metric,
        "threshold": METRICS[metric]["threshold0"], "tried": 0, "accepted": 0,
        "updated_at": 0.0, "baseline": None, "entries": [],
    }


class Archive:
    """One model's discoveries under one metric: ``index.json`` plus one PNG per entry."""

    def __init__(self, model_path: str, directory: Path | None = None,
                 metric: str = DEFAULT_METRIC):
        self.model_path = str(model_path)
        self.metric = metric if metric in METRICS else DEFAULT_METRIC
        self.dir = directory or archive_dir(model_path, self.metric)
        self.index = _empty_index(model_path, self.metric)
        self._matrix = None
        if self.dir and (self.dir / "index.json").exists():
            try:
                loaded = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
                self.index.update(loaded)
            except Exception as e:  # noqa: BLE001
                log.warning("could not read %s: %s", self.dir / "index.json", e)
            # An entry without its picture is no use to anyone; drop it here
            # rather than serve a broken card.
            self.index["entries"] = [e for e in self.index["entries"]
                                     if (self.dir / f"{e['id']}.png").exists()]
        self.index["metric"] = self.metric

    @classmethod
    def from_dir(cls, directory: Path) -> "Archive | None":
        f = directory / "index.json"
        if not f.exists():
            return None
        try:
            head = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        return cls(head.get("model_path", ""), directory, head.get("metric") or DEFAULT_METRIC)

    @property
    def entries(self) -> list[dict]:
        return self.index["entries"]

    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            rows = [e["embedding"] for e in self.entries]
            self._matrix = (np.asarray(rows, dtype=np.float32)
                            if rows else np.zeros((0, 1), dtype=np.float32))
        return self._matrix

    def save(self):
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index["updated_at"] = time.time()
        tmp = self.dir / "index.json.tmp"
        tmp.write_text(json.dumps(self.index), encoding="utf-8")
        os.replace(tmp, self.dir / "index.json")

    def novelty(self, vec: np.ndarray) -> float | None:
        m = self.matrix()
        if not len(m):
            return None
        d = 1.0 - m @ vec
        k = min(K, len(d))
        return float(np.sort(d)[:k].mean())

    def add(self, entry: dict, vec: np.ndarray):
        entry = {**entry, "embedding": [round(float(x), 4) for x in vec]}
        self.entries.append(entry)
        self.index["accepted"] = int(self.index.get("accepted", 0)) + 1
        self._matrix = None

    def find(self, did: str) -> dict | None:
        return next((e for e in self.entries if e["id"] == did), None)

    def remove(self, did: str) -> bool:
        e = self.find(did)
        if e is None:
            return False
        self.entries.remove(e)
        self._matrix = None
        try:
            (self.dir / f"{did}.png").unlink()
        except OSError:
            pass
        return True

    def prune(self):
        """Past the cap, the oldest non-starred entry makes room for the new one."""
        while len(self.entries) > CAP:
            victim = next((e for e in sorted(self.entries, key=lambda e: float(e.get("created_at", 0)))
                           if not e.get("starred")), None)
            if victim is None:
                return
            self.remove(victim["id"])

    def public_entries(self) -> list[dict]:
        missing = not model_exists(self.model_path)
        out = []
        for e in self.entries:
            pub = {k: v for k, v in e.items() if k != "embedding"}
            pub["image"] = str(self.dir / f"{e['id']}.png")
            pub["model_missing"] = missing
            pub["metric"] = self.metric
            out.append(pub)
        return out


def _archives(model_path: str | None = None) -> list[Archive]:
    """Archives to read: the worker's live one where it has one, disk otherwise.

    Routes and the worker must see the same object for the model being
    explored, or a star or delete written to disk is overwritten by the
    worker's next save of its own copy. A model has one archive per metric;
    all of them are listed.
    """
    root = archive_root()
    if not root.is_dir():
        return []
    key = previews.cache_key(model_path) if model_path else None
    if model_path and not key:
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if key and d.name != key and not d.name.startswith(f"{key}-"):
            continue
        a = Archive.from_dir(d)
        if a is None:
            continue
        out.append(explorer.live_archive(a.model_path, a.metric) or a)
    return out


def list_discoveries(model_path: str | None = None, since: float = 0.0,
                     sort: str = "newest", limit: int = 200) -> dict:
    entries = []
    for a in _archives(model_path):
        with explorer.archive_lock(a.model_path):
            pub = a.public_entries()
        entries.extend(e for e in pub if float(e.get("created_at", 0)) > since)
    if sort == "novel":
        entries.sort(key=lambda e: float(e.get("novelty", 0)), reverse=True)
    else:
        entries.sort(key=lambda e: float(e.get("created_at", 0)), reverse=True)
    total = len(entries)
    return {"entries": entries[:max(1, int(limit))], "total": total, "now": time.time()}


def _archive_holding(did: str, model_path: str | None) -> Archive | None:
    for a in _archives(model_path):
        if a.find(did):
            return a
    return None


def delete_discovery(did: str, model_path: str | None = None) -> bool:
    a = _archive_holding(did, model_path)
    if a is None:
        return False
    with explorer.archive_lock(a.model_path):
        a.remove(did)
        a.save()
    return True


def clear_discoveries(model_path: str | None = None) -> int:
    """Drop every discovery (of one model, or all), starred ones included.

    Saved presets are untouched: a star also wrote a Library entry, and that
    is the user's. The live archive keeps its baseline and starts its counts
    over; archives the worker does not hold are removed from disk entirely.
    """
    import shutil

    removed = 0
    for a in _archives(model_path):
        with explorer.archive_lock(a.model_path):
            removed += len(a.entries)
            if explorer.live_archive(a.model_path, a.metric) is a:
                for e in list(a.entries):
                    a.remove(e["id"])
                a.index.update({"tried": 0, "accepted": 0,
                                "threshold": METRICS[a.metric]["threshold0"]})
                a.save()
            elif a.dir and a.dir.is_dir():
                shutil.rmtree(a.dir, ignore_errors=True)
    return removed


def similar_discoveries(did: str, model_path: str | None = None, limit: int = 24) -> dict | None:
    """The archive-mates of one discovery, nearest first, each with its distance."""
    a = _archive_holding(did, model_path)
    if a is None:
        return None
    with explorer.archive_lock(a.model_path):
        anchor = a.find(did)
        vec = np.asarray(anchor["embedding"], dtype=np.float32)
        m = a.matrix()
        d = 1.0 - m @ vec
        pub = a.public_entries()
    ranked = sorted(zip(d.tolist(), pub), key=lambda t: t[0])
    out = []
    for dist, e in ranked:
        if e["id"] == did:
            continue
        out.append({**e, "distance": round(float(dist), 4)})
        if len(out) >= limit:
            break
    anchor_pub = next(e for e in pub if e["id"] == did)
    return {"anchor": anchor_pub, "entries": out}


def discovery_map(metric: str = DEFAULT_METRIC, model_path: str | None = None) -> dict:
    """A 2-D layout of every discovery under one metric: PCA of the embeddings.

    Embeddings from one metric are comparable across models, so with no
    ``model_path`` the map shows where different models' discoveries sit
    relative to each other. Coordinates are scaled to 0..1 with a margin.
    """
    entries, rows = [], []
    for a in _archives(model_path):
        if a.metric != metric:
            continue
        with explorer.archive_lock(a.model_path):
            pub = a.public_entries()
            rows.extend(e["embedding"] for e in a.entries)
        entries.extend(pub)
    n = len(rows)
    if n == 0:
        return {"metric": metric, "entries": []}
    if n < 3:
        pts = np.linspace(0.2, 0.8, n).reshape(-1, 1)
        xy = np.hstack([pts, np.full((n, 1), 0.5)])
    else:
        m = np.asarray(rows, dtype=np.float32)
        m = m - m.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        xy = m @ vt[:2].T
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
        xy = 0.06 + 0.88 * (xy - lo) / span
    for e, (x, y) in zip(entries, xy.tolist()):
        e["x"], e["y"] = round(float(x), 4), round(float(y), 4)
    return {"metric": metric, "entries": entries}


def star_discovery(did: str, model_path: str | None = None, starred: bool = True) -> dict | None:
    a = _archive_holding(did, model_path)
    if a is None:
        return None
    with explorer.archive_lock(a.model_path):
        e = a.find(did)
        e["starred"] = bool(starred)
        a.save()
    return {k: v for k, v in e.items() if k != "embedding"}


# --- candidates --------------------------------------------------------
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _rand_param(p: dict, rng: random.Random):
    kind = p.get("kind", "float")
    if kind == "select":
        opts = p.get("options") or [p.get("default")]
        return rng.choice(opts)
    lo, hi = p.get("min"), p.get("max")
    if lo is None or hi is None:
        return p.get("default")
    if kind == "int":
        step = int(p.get("step") or 1)
        n = int((hi - lo) // step)
        return int(lo + step * rng.randint(0, max(0, n)))
    return round(rng.uniform(float(lo), float(hi)), 4)


def _rand_group(rng: random.Random) -> str:
    return rng.choices(list(GROUP_WEIGHTS), list(GROUP_WEIGHTS.values()))[0]


def _window(sched: dict, rng: random.Random) -> tuple[float, float]:
    start = _clamp(float(sched.get("start", 0.0)) + rng.uniform(-0.15, 0.15))
    end = _clamp(float(sched.get("end", 1.0)) + rng.uniform(-0.15, 0.15))
    if end - start < 0.15:
        mid = (start + end) / 2
        start, end = _clamp(mid - 0.075), _clamp(mid + 0.075)
        if end - start < 0.15:
            start, end = (0.0, 0.15) if start < 0.5 else (0.85, 1.0)
    return round(start, 3), round(end, 3)


def _bend_id() -> str:
    return f"x-{uuid.uuid4().hex[:8]}"


def random_bend(rng: random.Random, catalog: list[dict]) -> dict:
    op = rng.choice(catalog)
    params = {p["name"]: _rand_param(p, rng) for p in op.get("params", [])}
    targets = [_rand_group(rng)]
    if rng.random() < 0.15:
        other = _rand_group(rng)
        if other not in targets:
            targets.append(other)
    start, end = _window(op.get("schedule") or {}, rng)
    return {"id": _bend_id(), "op": op["name"], "params": params, "targets": targets,
            "step_start": start, "step_end": end, "active": True}


def random_stack(rng: random.Random, catalog: list[dict]) -> list[dict]:
    n = rng.choices([s for s, _ in STACK_SIZES], [w for _, w in STACK_SIZES])[0]
    return [random_bend(rng, catalog) for _ in range(n)]


def mutate(stack: list[dict], rng: random.Random, catalog: list[dict]) -> list[dict]:
    """One change to a copy of ``stack``; the copy gets fresh ids."""
    by_name = {o["name"]: o for o in catalog}
    out = [{**b, "params": dict(b.get("params") or {}), "targets": list(b.get("targets") or []),
            "id": _bend_id()} for b in stack]
    moves = ["amount", "target", "window", "op"]
    if len(out) < MAX_BENDS:
        moves.append("add")
    if len(out) > 1:
        moves.append("remove")
    move = rng.choice(moves)
    b = rng.choice(out)
    op = by_name.get(b["op"])
    if move == "amount" and op and op.get("params"):
        key = op.get("amount_param") or op["params"][0]["name"]
        spec = next((p for p in op["params"] if p["name"] == key), None)
        if spec and spec.get("kind", "float") != "select" and spec.get("min") is not None:
            lo, hi = float(spec["min"]), float(spec["max"])
            v = float(b["params"].get(key, spec.get("default") or 0.0))
            if key == op.get("amount_param") and v != 0:
                v = v * rng.uniform(0.5, 2.0)
            else:
                v = v + rng.uniform(-0.2, 0.2) * (hi - lo)
            v = max(lo, min(hi, v))
            b["params"][key] = int(round(v)) if spec.get("kind") == "int" else round(v, 4)
        else:
            move = "target"
    if move == "target":
        b["targets"] = [_rand_group(rng)]
    elif move == "window":
        d = rng.uniform(-0.1, 0.1)
        s, e = _clamp(b["step_start"] + d), _clamp(b["step_end"] + d)
        if e - s >= 0.15:
            b["step_start"], b["step_end"] = round(s, 3), round(e, 3)
    elif move == "op" and op:
        same = [o for o in catalog if o["category"] == op["category"] and o["name"] != op["name"]]
        if same:
            new = rng.choice(same)
            b["op"] = new["name"]
            b["params"] = {p["name"]: p.get("default") for p in new.get("params", [])}
    elif move == "add":
        out.append(random_bend(rng, catalog))
    elif move == "remove":
        out.remove(b)
    return out


def _tournament(entries: list[dict], rng: random.Random) -> dict:
    a, b = rng.choice(entries), rng.choice(entries)
    return a if float(a.get("novelty", 0)) >= float(b.get("novelty", 0)) else b


# --- rendering and embedding ------------------------------------------
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _Dino:
    """Lazily-loaded DINOv2 ViT-B/14, through torch.hub.

    The first use downloads the hub repo and the checkpoint (about 350 MB)
    into torch's hub cache; after that it loads offline. Self-supervised
    features, no text: they separate images by structure and texture rather
    than by what CLIP would caption them as, which is a different idea of
    "looks new" and the reason the metric is a choice.
    """

    _inst = None

    @classmethod
    def get(cls, device):
        import torch

        if cls._inst is None:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
            cls._inst = model.eval()
        model = cls._inst
        if next(model.parameters()).device.type != torch.device(device).type:
            model = model.to(device)
            cls._inst = model
        return model


def _batch(images: list[Image.Image], mean, std):
    import torch

    arrs = [np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC),
                       dtype=np.float32) / 255.0 for im in images]
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def embed(images: list[Image.Image], device: str, metric: str = DEFAULT_METRIC) -> np.ndarray:
    """Unit-length embedding of the *average* of ``images`` under ``metric``.

    CLIP: ViT-B/32 image features. ``_Clip.get`` drops CLIP's own preprocess,
    so the resize and normalisation are done here to match it. DINOv2:
    ViT-B/14 CLS features with ImageNet normalisation.
    """
    import torch

    if metric == "dinov2":
        model = _Dino.get(device)
        x = _batch(images, _IMAGENET_MEAN, _IMAGENET_STD)
        x = x.to(device=device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            e = model(x).float()
    else:
        model, _ = _Clip.get(device)
        x = _batch(images, _CLIP_MEAN, _CLIP_STD)
        x = x.to(device=device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            e = model.encode_image(x).float()
    e = e / e.norm(dim=-1, keepdim=True)
    v = e.mean(0)
    v = v / v.norm()
    return v.detach().cpu().numpy().astype(np.float32)


def _degenerate(images: list[Image.Image]) -> bool:
    for im in images:
        a = np.asarray(im.convert("L"), dtype=np.float32)
        if not np.isfinite(a).all() or a.std() < 2.0:
            return True
    return False


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


# --- the worker --------------------------------------------------------
class Explorer:
    def __init__(self):
        self._lock = threading.Lock()
        self._archive_locks: dict[str, threading.Lock] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._archive: Archive | None = None
        self._state = self._idle_state()

    @staticmethod
    def _idle_state() -> dict:
        return {"running": False, "model_path": None, "model_name": None,
                "metric": DEFAULT_METRIC, "tried": 0, "accepted": 0, "archive_size": 0,
                "threshold": METRICS[DEFAULT_METRIC]["threshold0"],
                "last_novelty": None, "yielding_to": None, "error": None}

    # Archive files are rewritten by the worker and by delete/star from routes.
    def archive_lock(self, model_path: str) -> threading.Lock:
        with self._lock:
            return self._archive_locks.setdefault(str(model_path), threading.Lock())

    def live_archive(self, model_path: str, metric: str | None = None) -> Archive | None:
        """The worker's in-memory archive for ``model_path`` (and ``metric``), if it holds one."""
        with self._lock:
            a = self._archive
        if a is None or a.model_path != str(model_path):
            return None
        if metric is not None and a.metric != metric:
            return None
        return a

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    def start(self, model_path: str, metric: str = DEFAULT_METRIC) -> dict:
        if metric not in METRICS:
            raise ValueError(f"unknown novelty metric: {metric}")
        self.stop()
        with self._lock:
            self._state = {**self._idle_state(), "running": True, "metric": metric,
                           "model_path": str(model_path), "model_name": model_name(model_path),
                           "threshold": METRICS[metric]["threshold0"]}
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(str(model_path), metric),
                                            name="explore", daemon=True)
            self._thread.start()
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            t, stop = self._thread, self._stop
        if t is not None and t.is_alive():
            stop.set()
            t.join(timeout=10.0)
        self._set(running=False, yielding_to=None)
        return self.status()

    def _yield_to(self) -> str | None:
        for j in registry.list():
            if j.get("status") == "running" and j.get("kind") in YIELD_KINDS:
                return j["kind"]
        return None

    def _render(self, model_path: str, runtime, stop: threading.Event) -> list[Image.Image]:
        params = SampleParams(
            model_path=model_path, ema=True, image_size=SIZE, steps=STEPS,
            sampler=SAMPLER, seed=FIRST_SEED, batch_size=SEEDS, postproc={},
        )
        last = None
        for frame in sampler.run(params, None, None, runtime, cancel=stop.is_set):
            last = frame
        if last is None or stop.is_set():
            return []
        return list(last.get("images_pp") or [last["image_pp"]])

    def _current_archive(self, model_path: str, metric: str = DEFAULT_METRIC) -> Archive:
        with self._lock:
            a = self._archive
        if a is None or a.model_path != model_path or a.metric != metric:
            a = Archive(model_path, metric=metric)
            with self._lock:
                self._archive = a
        return a

    def _run(self, model_path: str, metric: str = DEFAULT_METRIC):
        stop = self._stop
        spec = METRICS[metric]
        try:
            device = pick_device("auto")
            bundle = manager.load(model_path, device=device, ema=True)
            meta, backend = bundle["meta"], bundle["backend"]
            if not backend.capabilities.bend:
                self._set(running=False, error="this model cannot be bent")
                return
            arch = self._current_archive(model_path, metric)
            arch.index["model"] = {
                "name": model_name(model_path), "path": model_path,
                "key": previews.cache_key(model_path),
                "mtype": getattr(meta, "mtype", None), "step": getattr(meta, "step", None),
            }
            catalog = ops_mod.catalog()
            rng = random.Random()
            self._set(tried=arch.index.get("tried", 0), accepted=len(arch.entries),
                      archive_size=len(arch.entries), threshold=arch.index["threshold"])

            if arch.index.get("baseline") is None:
                imgs = self._render(model_path, None, stop)
                if not imgs:
                    return
                with self.archive_lock(model_path):
                    arch.index["baseline"] = [round(float(x), 4)
                                              for x in embed(imgs, device, metric)]
                    arch.save()
            base = np.asarray(arch.index["baseline"], dtype=np.float32)
            # The threshold adapts once per window of tries, not per try:
            # a burst of early accepts would otherwise compound the step
            # twenty times over and pin it to the ceiling.
            window_tries = window_accepts = 0
            since_save = 0

            while not stop.is_set():
                y = self._yield_to()
                if y:
                    self._set(yielding_to=y)
                    stop.wait(1.0)
                    continue
                self._set(yielding_to=None, error=None)

                if arch.entries and rng.random() < 0.5:
                    parent = _tournament(arch.entries, rng)
                    stack, source = mutate(parent["bends"], rng, catalog), f"mutate:{parent['id']}"
                else:
                    stack, source = random_stack(rng, catalog), "random"

                try:
                    runtime = bending.build_runtime(stack, meta, backend=backend)
                    imgs = self._render(model_path, runtime, stop)
                except Exception as e:  # noqa: BLE001
                    if _is_oom(e):
                        self._set(error="GPU out of memory; trying again in 30 s")
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass
                        stop.wait(30.0)
                        continue
                    log.warning("candidate failed (%s): %s", source, e)
                    arch.index["tried"] = int(arch.index.get("tried", 0)) + 1
                    continue
                if stop.is_set():
                    break
                arch.index["tried"] = int(arch.index.get("tried", 0)) + 1
                if not imgs or _degenerate(imgs):
                    self._set(tried=arch.index["tried"])
                    continue

                vec = embed(imgs, device, metric)
                dist_base = float(1.0 - base @ vec)
                nov = arch.novelty(vec)
                if nov is None:
                    nov = dist_base
                thr = float(arch.index["threshold"])
                bootstrap = len(arch.entries) < K
                # Every phase demands the floor distance from what is already
                # kept (with nothing kept, that is the baseline). Bootstrap
                # only waives the adaptive threshold; without the floor it
                # let near-duplicates of the first entries in.
                accept = (
                    dist_base > spec["baseline_eps"]
                    and nov > spec["min_distance"]
                    and (bootstrap or nov > thr or rng.random() < INJECT_P)
                )
                if accept and nov <= thr and len(arch.entries) >= K:
                    source = "inject"

                if accept:
                    did = uuid.uuid4().hex[:10]
                    params = SampleParams(
                        model_path=model_path, ema=True, image_size=SIZE, steps=STEPS,
                        sampler=SAMPLER, seed=FIRST_SEED, batch_size=1, postproc={},
                    )
                    card = build_card(
                        params, model_path=model_path, model_name=model_name(model_path),
                        bends=stack, kind="discovery",
                        extra={"discovery_id": did, "novelty": round(nov, 4),
                               "seeds": [FIRST_SEED + i for i in range(SEEDS)],
                               "source": source, "metric": metric},
                    )
                    entry = {
                        "id": did, "seed": FIRST_SEED, "bends": stack,
                        "novelty": round(nov, 4), "created_at": time.time(),
                        "source": source, "starred": False, "metric": metric,
                        "model_path": model_path, "model_name": model_name(model_path),
                    }
                    with self.archive_lock(model_path):
                        arch.dir.mkdir(parents=True, exist_ok=True)
                        save_with_params(imgs[0], arch.dir / f"{did}.png", card)
                        arch.add(entry, vec)
                        arch.prune()
                        arch.save()
                    since_save = 0

                if not bootstrap:
                    window_tries += 1
                    window_accepts += int(accept)
                if window_tries >= WINDOW:
                    if window_accepts > WINDOW_HIGH:
                        thr *= 1.1
                    elif window_accepts == 0:
                        thr *= 0.8
                    window_tries = window_accepts = 0
                thr = max(spec["threshold"][0], min(spec["threshold"][1], thr))
                arch.index["threshold"] = round(thr, 4)
                since_save += 1
                if since_save >= 20:
                    with self.archive_lock(model_path):
                        arch.save()
                    since_save = 0

                self._set(tried=arch.index["tried"], accepted=arch.index.get("accepted", 0),
                          archive_size=len(arch.entries), threshold=round(thr, 4),
                          last_novelty=round(nov, 4))
        except Exception as e:  # noqa: BLE001
            log.exception("explorer stopped: %s", e)
            self._set(error=str(e))
        finally:
            try:
                with self.archive_lock(model_path):
                    self._current_archive(model_path, metric).save()
            except Exception:  # noqa: BLE001
                pass
            self._set(running=False, yielding_to=None)


explorer = Explorer()
