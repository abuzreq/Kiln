"""What the last model scan found, kept on disk so a launch need not read it all again.

Listing models used to read every checkpoint in full on every launch -- and to
read one, import torch: 152 models took about 19 s before the Canvas picker or
the home page could show the finished list. The backends' in-memory cache only
helped within one run. This index keeps, per file, the descriptor a backend
made from it and the file's signature (mtime and size, the same test the
in-memory cache uses); a file whose signature still matches is listed from here
without loading any backend, so torch is not imported just to list models.

Things that turned out not to be models (a folder with no Diffusers config, a
.pt Kiln could not read) are kept too, with no descriptor, so they do not send
every launch back to the backends. A changed file has a new signature and is
read again.
"""
import json
import os
from dataclasses import asdict, fields
from pathlib import Path

from app.core.backends.base import ModelDescriptor

VERSION = 1
_FIELDS = {f.name for f in fields(ModelDescriptor)}


def _file() -> Path:
    from app.core.config import workspace

    return workspace.cache / "model-index.json"


def signature(path: Path) -> "list | None":
    """mtime and size of a checkpoint; for a model folder, of its top-level files."""
    try:
        st = path.stat()
        if path.is_file():
            return [st.st_mtime, st.st_size]
        sig = [st.st_mtime]
        for f in sorted(path.iterdir()):
            if f.is_file():
                fs = f.stat()
                sig += [f.name, fs.st_mtime, fs.st_size]
        return sig
    except OSError:
        return None


def candidates(sources: "list[tuple[Path, str]]") -> "list[tuple[Path, str]]":
    """Everything in the scanned folders a backend might call a model: each
    checkpoint file and each folder. Kiln's own ``.kiln`` and other dot-folders
    hold no models."""
    out = []
    for d, label in sources:
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for p in entries:
            if p.name.startswith("."):
                continue
            try:
                if (p.suffix == ".pt" and p.is_file()) or p.is_dir():
                    out.append((p, label))
            except OSError:
                continue
    return out


def load() -> dict:
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def save(entries: dict) -> None:
    f = _file()
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"version": VERSION, "entries": entries}, default=str),
                       encoding="utf-8")
        os.replace(tmp, f)
    except OSError:
        pass  # only a head start for the next launch


def entry(sig: list, meta: "ModelDescriptor | None") -> dict:
    if meta is None:
        return {"sig": sig, "meta": None}
    m = asdict(meta)
    m.pop("thumbnail", None)  # worked out fresh each time: one can be added later
    m.pop("source", None)     # depends on which folder listed it
    return {"sig": sig, "meta": m}


def descriptor(meta: dict) -> ModelDescriptor:
    return ModelDescriptor(**{k: v for k, v in meta.items() if k in _FIELDS})
