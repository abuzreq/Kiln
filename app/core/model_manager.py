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

    # --- discovery ----------------------------------------------------
    def _sources(self, extra_dirs: list[Path] | None = None) -> "list[tuple[Path, str]]":
        """Where models live, in priority order. App policy, not engine policy."""
        from app.core.config import workspace

        repo_root = Path(__file__).resolve().parents[2]
        sources: list[tuple[Path, str]] = [
            (workspace.models, "workspace"),
            (repo_root / "models" / "pretrained", "pretrained"),
            (repo_root / "models" / "fine_tuned", "fine-tuned"),
            (repo_root / "vendor" / "xurdif" / "models", "vendored"),
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
        sources = self._sources(extra_dirs)
        out: list[ModelDescriptor] = []
        seen: set[str] = set()
        for name in backends.available():
            try:
                found = backends.get(name).scan(sources)
            except Exception as e:  # noqa: BLE001
                log.warning("backend %s failed to scan: %s", name, e)
                continue
            for meta in found:
                if meta.path in seen:
                    continue
                seen.add(meta.path)
                out.append(meta)
        return out

    def scan_public(self) -> list[dict]:
        """Scan plus role (main vs training checkpoint) and starred flags."""
        from app.core import library

        stars = {library._norm_star_path(p) for p in library.list_stars()}
        out = []
        for m in self.scan():
            d = m.to_dict()
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
            mtime = Path(m.path).stat().st_mtime
            if card and card.get("created_at"):
                d["created_at"] = card["created_at"]
            d["mtime"] = card.get("created_at") if card and card.get("created_at") else mtime
            out.append(d)
        return out

    # --- loading ------------------------------------------------------
    def load(self, path: str, device: str = "cpu", ema: bool = True):
        backend, ref = backends.resolve(str(path))
        key = f"{ref}|{device}|{'ema' if ema else 'model'}"
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        model, meta = backend.load(ref, device=device, ema=ema)

        bundle = {"model": model, "meta": meta, "backend": backend, "ref": ref}
        with self._lock:
            self._cache[key] = bundle
            self._cache.move_to_end(key)
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


def _free_cuda():
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


manager = ModelManager()
