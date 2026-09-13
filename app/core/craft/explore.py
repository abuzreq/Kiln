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
CAP = 200                  # entries per model; the densest non-starred one goes
THRESHOLD0 = 0.08
THRESHOLD_MIN, THRESHOLD_MAX = 0.02, 0.4
WINDOW = 20                # tries per threshold adaptation
WINDOW_HIGH = 4            # more accepts than this in a window: raise the bar
INJECT_P = 0.05            # chance to keep a merely-different candidate anyway
MIN_DISTANCE = 0.03        # below this from the baseline, a bend did nothing worth keeping
BASELINE_EPS = 0.01        # below this, a bend did nothing at all
MAX_BENDS = 3

YIELD_KINDS = ("sample", "inpaint", "randomize", "bend_sweep", "sweep")
GROUP_WEIGHTS = {"encoder": 0.25, "mid": 0.25, "decoder": 0.25,
                 "attention": 0.1, "blocks": 0.05, "all": 0.05}
STACK_SIZES = ((1, 0.5), (2, 0.35), (3, 0.15))


# --- archive -----------------------------------------------------------
def archive_root() -> Path:
    return workspace.cache / "discoveries"


def archive_dir(model_path: str | Path) -> Path | None:
    key = previews.cache_key(model_path)
    return archive_root() / key if key else None


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


def _empty_index(model_path: str) -> dict:
    return {
        "version": 1, "model_path": str(model_path), "model": {},
        "threshold": THRESHOLD0, "tried": 0, "accepted": 0,
        "updated_at": 0.0, "baseline": None, "entries": [],
    }


class Archive:
    """One model's discoveries: ``index.json`` plus one PNG per entry."""

    def __init__(self, model_path: str, directory: Path | None = None):
        self.model_path = str(model_path)
        self.dir = directory or archive_dir(model_path)
        self.index = _empty_index(model_path)
        self._matrix = None
        if self.dir and (self.dir / "index.json").exists():
            try:
                loaded = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
                self.index.update(loaded)
            except Exception as e:  # noqa: BLE001
                log.warning("could not read %s: %s", self.dir / "index.json", e)

    @classmethod
    def from_dir(cls, directory: Path) -> "Archive | None":
        f = directory / "index.json"
        if not f.exists():
            return None
        try:
            model_path = json.loads(f.read_text(encoding="utf-8")).get("model_path", "")
        except Exception:  # noqa: BLE001
            return None
        return cls(model_path, directory)

    @property
    def entries(self) -> list[dict]:
        return self.index["entries"]

    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            rows = [e["embedding"] for e in self.entries]
            self._matrix = (np.asarray(rows, dtype=np.float32)
                            if rows else np.zeros((0, 512), dtype=np.float32))
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
        """Drop the densest non-starred entries until the cap holds."""
        while len(self.entries) > CAP:
            m = self.matrix()
            d = 1.0 - m @ m.T
            np.fill_diagonal(d, np.inf)
            k = min(K, len(m) - 1)
            density = np.sort(d, axis=1)[:, :k].mean(axis=1)
            order = np.argsort(density)
            victim = next((self.entries[i] for i in order
                           if not self.entries[i].get("starred")), None)
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
            out.append(pub)
        return out


def _archives(model_path: str | None = None) -> list[Archive]:
    """Archives to read: the worker's live one where it has one, disk otherwise.

    Routes and the worker must see the same object for the model being
    explored, or a star or delete written to disk is overwritten by the
    worker's next save of its own copy.
    """
    if model_path:
        live = explorer.live_archive(model_path)
        if live is not None:
            return [live]
        d = archive_dir(model_path)
        return [Archive(model_path, d)] if d and (d / "index.json").exists() else []
    root = archive_root()
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        a = Archive.from_dir(d) if d.is_dir() else None
        if a is None:
            continue
        out.append(explorer.live_archive(a.model_path) or a)
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


def embed(images: list[Image.Image], device: str) -> np.ndarray:
    """Unit-length CLIP ViT-B/32 embedding of the *average* of ``images``.

    ``_Clip.get`` drops CLIP's own preprocess, so the resize and normalisation
    are done here to match it.
    """
    import torch

    model, _ = _Clip.get(device)
    arrs = [np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC),
                       dtype=np.float32) / 255.0 for im in images]
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
    x = ((x - mean) / std).to(device=device, dtype=next(model.parameters()).dtype)
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
                "tried": 0, "accepted": 0, "archive_size": 0, "threshold": THRESHOLD0,
                "last_novelty": None, "yielding_to": None, "error": None}

    # Archive files are rewritten by the worker and by delete/star from routes.
    def archive_lock(self, model_path: str) -> threading.Lock:
        with self._lock:
            return self._archive_locks.setdefault(str(model_path), threading.Lock())

    def live_archive(self, model_path: str) -> Archive | None:
        """The worker's in-memory archive for ``model_path``, if it holds one."""
        with self._lock:
            a = self._archive
        return a if a is not None and a.model_path == str(model_path) else None

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    def start(self, model_path: str) -> dict:
        self.stop()
        with self._lock:
            self._state = {**self._idle_state(), "running": True,
                           "model_path": str(model_path), "model_name": model_name(model_path)}
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(str(model_path),),
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

    def _current_archive(self, model_path: str) -> Archive:
        with self._lock:
            a = self._archive
        if a is None or a.model_path != model_path:
            a = Archive(model_path)
            with self._lock:
                self._archive = a
        return a

    def _run(self, model_path: str):
        stop = self._stop
        try:
            device = pick_device("auto")
            bundle = manager.load(model_path, device=device, ema=True)
            meta, backend = bundle["meta"], bundle["backend"]
            if not backend.capabilities.bend:
                self._set(running=False, error="this model cannot be bent")
                return
            arch = self._current_archive(model_path)
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
                    arch.index["baseline"] = [round(float(x), 4) for x in embed(imgs, device)]
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

                vec = embed(imgs, device)
                dist_base = float(1.0 - base @ vec)
                nov = arch.novelty(vec)
                if nov is None:
                    nov = dist_base
                thr = float(arch.index["threshold"])
                bootstrap = len(arch.entries) < K
                accept = dist_base > BASELINE_EPS and (
                    (bootstrap and dist_base > MIN_DISTANCE)
                    or nov > thr
                    or (nov > MIN_DISTANCE and rng.random() < INJECT_P)
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
                               "source": source},
                    )
                    entry = {
                        "id": did, "seed": FIRST_SEED, "bends": stack,
                        "novelty": round(nov, 4), "created_at": time.time(),
                        "source": source, "starred": False,
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
                thr = max(THRESHOLD_MIN, min(THRESHOLD_MAX, thr))
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
                    self._current_archive(model_path).save()
            except Exception:  # noqa: BLE001
                pass
            self._set(running=False, yielding_to=None)


explorer = Explorer()
