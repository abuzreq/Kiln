"""Discover, load and cache models, whichever backend owns them.

This module used to also *be* the xurdif loader. That part now lives in
``app.core.backends.xurdif``; what stays here is the engine-independent work:
knowing where in the workspace models live, and keeping a bounded LRU of loaded
nets so trying models out does not simply grow VRAM forever.

``read_meta`` and ``manager.load`` keep their old signatures and return shapes so
existing callers (routes, sampler, craft, the smoke scripts) are untouched.
"""
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from app.core import backends
from app.core.backends.base import ModelDescriptor
from utils.logger import get_logger

log = get_logger("model_manager")

# Kept as an alias: the old name is descriptive at call sites that only want
# metadata, and third-party/dev scripts may still import it.
ModelMeta = ModelDescriptor


def _torch():
    import torch  # deferred so the app boots without a torch install

    return torch


def read_meta(path: str | Path) -> ModelDescriptor:
    """Metadata for one model, from whichever backend recognises it."""
    backend, ref = backends.resolve(str(path))
    return backend.describe(ref)


_REPO_ROOT = Path(__file__).resolve().parents[2]
# The install's own model folders. Models used to be kept here (xurdif's
# convention), so they are still scanned for anyone who has files in them, but
# the repository no longer ships them: the workspace is where models live.
LEGACY_MODEL_DIRS = (
    (_REPO_ROOT / "models" / "pretrained", "pretrained"),
    (_REPO_ROOT / "models" / "fine_tuned", "fine-tuned"),
    (_REPO_ROOT / "vendor" / "xurdif" / "models", "vendored"),
)


# The tails of those folders' paths, matched wherever the install was: recipes
# record absolute paths, and the install recorded may be another checkout, or
# this one before it moved.
_LEGACY_TAILS = (("models", "pretrained"), ("models", "fine_tuned"), ("vendor", "xurdif", "models"))


def _relocated(path: Path) -> Path | None:
    """A checkpoint moved from an install's model folders into the workspace.

    Only for paths in those legacy folders: a missing file anywhere else is
    missing, and matching it to some other model of the same name would quietly
    sample the wrong one. Mirrors relocateModelPath in the frontend.
    """
    from app.core.config import workspace

    parts = tuple(x.lower() for x in Path(str(path).replace("\\", "/")).parent.parts)
    if not any(parts[-len(t):] == t for t in _LEGACY_TAILS):
        return None
    moved = workspace.models / path.name
    return moved if moved.is_file() else None


backends.set_relocator(_relocated)


def _thumbnail(path: Path) -> str | None:
    """A listed model's picture, by the rule its backend uses when it reads one.

    For a model listed from the index, which no backend read this time: a
    Diffusers folder's own ``thumbnail.png``; for a checkpoint, a run's
    ``sample-N.png`` beside ``model-N.pt``, else its own picture.
    """
    if path.is_dir():
        t = path / "thumbnail.png"
        return str(t) if t.is_file() else None
    if path.stem.startswith("model-"):
        sample = path.with_name("sample-" + path.stem.split("-", 1)[1] + ".png")
        if sample.is_file():
            return str(sample)
    from app.core import library

    own = library.own_thumb(path)
    return str(own) if own else None


def _sidecar_thumbnail(path: Path) -> str | None:
    """Back-compat shim; the rule is xurdif's and lives with it now."""
    from app.core.backends.xurdif import loader

    return loader.sidecar_thumbnail(Path(path))


class ModelManager:
    """Caches loaded nets keyed by (ref, device, ema), most-recent first.

    Bounded on purpose: every cached bundle pins a net on the GPU, so an
    unbounded cache meant VRAM only ever grew as you tried models out.
    """

    MAX_LOADED = 2

    def __init__(self):
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._lock = threading.Lock()
        # One gate per cache key while it is being loaded, so runs that miss on
        # the same model at once wait for one load instead of each reading the
        # file and building a second copy on the device.
        self._loading: dict[str, threading.Lock] = {}

    # --- discovery ----------------------------------------------------
    def _sources(self, extra_dirs: list[Path] | None = None) -> "list[tuple[Path, str]]":
        """Where models live, in priority order. App policy, not engine policy."""
        from app.core.config import workspace

        sources: list[tuple[Path, str]] = [
            (workspace.models, "workspace"),
            *LEGACY_MODEL_DIRS,   # silently, and only if they exist (scan skips missing)
        ]
        # workspace runs + any leftover per-project run folders
        runs = workspace.runs
        if runs.exists():
            for r in runs.iterdir():
                if r.is_dir():
                    sources.append((r, f"run:{r.name}"))
        if workspace.projects.exists():
            for proj in workspace.projects.iterdir():
                if proj.is_dir():
                    mdir = proj / "models"
                    if mdir.exists():
                        sources.append((mdir, "workspace"))
                    pruns = proj / "runs"
                    if pruns.exists():
                        for r in pruns.iterdir():
                            if r.is_dir():
                                sources.append((r, f"run:{r.name}"))
        if extra_dirs:
            sources.extend((d, "custom") for d in extra_dirs)
        return sources

    def scan(self, extra_dirs: list[Path] | None = None) -> list[ModelDescriptor]:
        return list(self.iter_scan(extra_dirs))

    def iter_scan(self, extra_dirs: list[Path] | None = None):
        """``scan``, yielding each model as soon as it is known.

        Files the on-disk index (``model_index``) already knows, unchanged, come
        first and need no backend. Only the rest go to the backends, which read
        them -- and load torch to do it.
        """
        from app.core import model_index

        sources = self._sources(extra_dirs)
        known = model_index.load()
        fresh: dict = {}
        unread: list[tuple[str, list | None]] = []
        seen: set[str] = set()
        for path, label in model_index.candidates(sources):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            sig = model_index.signature(path)
            hit = known.get(key)
            if sig is not None and isinstance(hit, dict) and hit.get("sig") == sig:
                fresh[key] = hit
                if hit.get("meta"):
                    meta = model_index.descriptor(hit["meta"])
                    meta.source = label
                    meta.thumbnail = _thumbnail(path)
                    yield meta
            else:
                unread.append((key, sig))

        if unread:
            found: dict[str, ModelDescriptor] = {}
            failed = False
            for name in backends.available():
                try:
                    for meta in backends.get(name).iter_scan(sources, skip=set(fresh) | set(found)):
                        if meta.path in found:
                            continue
                        found[meta.path] = meta
                        yield meta
                except Exception as e:  # noqa: BLE001
                    # Models already yielded stay listed; the rest of this backend's
                    # are lost, as they were when scan() raised before yielding.
                    log.warning("backend %s failed to scan: %s", name, e)
                    failed = True
            for key, sig in unread:
                if sig is None:
                    continue
                if key in found:
                    fresh[key] = model_index.entry(sig, found[key])
                elif not failed:
                    # Not a model, as far as every backend could tell. After a
                    # failure it may be one the failed backend never reached, so
                    # it is left out and read again next time.
                    fresh[key] = model_index.entry(sig, None)
        if fresh != known:
            model_index.save(fresh)

    def scan_public(self, include_hidden: bool = False) -> list[dict]:
        """Scan plus role (main vs training checkpoint), starred and ownership flags.

        ``owned`` says whether Kiln may delete the file (see
        ``library.owned_by_kiln``); the UI offers Hide for everything else.
        Hidden models are left out unless ``include_hidden``.
        """
        return list(self.iter_public(include_hidden))

    def iter_public(self, include_hidden: bool = False):
        """``scan_public``, one model at a time, for the streamed listing."""
        from app.core import library

        stars = {library._norm_star_path(p) for p in library.list_stars()}
        hidden = set(library.list_hidden())
        for m in self.iter_scan():
            is_hidden = library._norm_star_path(m.path) in hidden
            if is_hidden and not include_hidden:
                continue
            d = m.to_dict()
            d["hidden"] = is_hidden
            d["owned"] = Path(m.path).is_file() and library.owned_by_kiln(m.path)
            is_ckpt = (m.source or "").startswith("run:")
            d["role"] = "checkpoint" if is_ckpt else "main"
            d["group"] = (m.source or "")[4:] if is_ckpt else None
            d["starred"] = library._norm_star_path(m.path) in stars
            card = library.read_card(m.path)
            if card:
                d["name"] = card.get("name") or d["name"]
                d["original_name"] = card.get("original_name") or d["name"]
                d["trained_as"] = card.get("trained_as") or []
                if d.get("step") is None and card.get("step") is not None:
                    d["step"] = card["step"]
            else:
                d["original_name"] = d["name"]
                d["trained_as"] = []
            d["renamable"] = library.can_rename(m.path)
            try:
                mtime = Path(m.path).stat().st_mtime
            except OSError:
                # Deleted between the scan and here. One vanished model must not
                # take the whole listing down with it.
                continue
            if card and card.get("created_at"):
                d["created_at"] = card["created_at"]
            d["mtime"] = card.get("created_at") if card and card.get("created_at") else mtime
            yield d

    def skipped(self, extra_dirs: "list[Path] | None" = None) -> list[dict]:
        """Files that look like models but could not be read, and why.

        A checkpoint Kiln cannot parse is skipped with only a log line, so a
        dropped-in file of the wrong kind simply never appears -- no error, no
        entry, nothing to act on. This is what the Models screen shows instead.
        """
        from app.core.backends.xurdif.loader import describe

        known = {str(Path(m.path).resolve()) for m in self.scan()}
        out: list[dict] = []
        for d, label in self._sources(extra_dirs):
            if not d.exists():
                continue
            for pt in sorted(d.glob("*.pt")):
                try:
                    if str(pt.resolve()) in known:
                        continue
                except OSError:
                    continue
                try:
                    describe(pt)
                except Exception as e:  # noqa: BLE001
                    out.append({"path": str(pt), "name": pt.name, "source": label,
                                "why": str(e) or type(e).__name__})
        return out

    # --- loading ------------------------------------------------------
    def describe(self, path: str):
        """``(descriptor, backend)`` for a model, without loading its weights.

        For callers that need to know what a model is -- its mults, whether it
        can be bent -- and not to run it. Loading for that put a second, CPU
        copy of every generated-from model in the two-slot cache.
        """
        backend, ref = backends.resolve(str(path))
        return backend.describe(ref), backend

    def load(self, path: str, device: str = "cpu", ema: bool = True):
        backend, ref = backends.resolve(str(path))
        key = f"{ref}|{device}|{'ema' if ema else 'model'}"
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            gate = self._loading.setdefault(key, threading.Lock())

        with gate:
            with self._lock:
                if key in self._cache:          # loaded while we waited
                    self._cache.move_to_end(key)
                    return self._cache[key]

            with _on_default_stream(device):
                model, meta = backend.load(ref, device=device, ema=ema)

            bundle = {"model": model, "meta": meta, "backend": backend, "ref": ref}
            with self._lock:
                self._cache[key] = bundle
                self._cache.move_to_end(key)
                self._loading.pop(key, None)
                evicted = False
                while len(self._cache) > self.MAX_LOADED:
                    old_key, _ = self._cache.popitem(last=False)
                    log.info("evicting cached model %s", old_key)
                    evicted = True
        if evicted:
            _free_cuda()
        return bundle

    def evict(self, path: str):
        """Drop every cached bundle for one model (e.g. a stale merge preview)."""
        try:
            ident = str(backends.parse_ref(str(path)))
        except Exception:  # noqa: BLE001
            ident = str(path)
        with self._lock:
            keys = [k for k in self._cache if k.rsplit("|", 2)[0] == ident]
            for k in keys:
                del self._cache[k]
        if keys:
            _free_cuda()
        return len(keys)

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
        _free_cuda()


@contextmanager
def _on_default_stream(device: str):
    """Load onto the GPU on the default stream, and finish before returning.

    Generation lanes run on CUDA streams of their own. Weights copied up on one
    lane's stream would be read by another lane's stream with nothing ordering
    the copy before the read, and would live in that one lane's allocator pool.
    """
    torch = _torch()
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        yield
        return
    stream = torch.cuda.default_stream(torch.device(device))
    with torch.cuda.stream(stream):
        yield
    stream.synchronize()


def _free_cuda():
    from app.core import devices

    devices.empty_cache()


manager = ModelManager()
