"""The Library: persistent, reusable crafting assets.

Stores bend presets, merge recipes, and sampling presets as JSON files under the
workspace ``library/`` tree so they can be reapplied across models and projects.
"""
import json
import os
import time
from pathlib import Path

from app.core.config import workspace
from utils.exceptions import NotFoundError, ValidationError
from utils.validators import safe_name


def _dir_for(kind: str) -> Path:
    return {
        "bends": workspace.bends,
        "recipes": workspace.recipes,
        "presets": workspace.presets,
    }[kind]


def save_entry(kind: str, name: str, payload: dict) -> dict:
    name = safe_name(name, f"{kind[:-1]} name")
    d = _dir_for(kind)
    d.mkdir(parents=True, exist_ok=True)
    entry = dict(payload)
    entry["name"] = name
    entry.setdefault("created_at", time.time())
    entry["updated_at"] = time.time()
    (d / f"{name}.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return entry


def list_entries(kind: str) -> list[dict]:
    d = _dir_for(kind)
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out


def get_entry(kind: str, name: str) -> dict:
    f = _dir_for(kind) / f"{name}.json"
    if not f.exists():
        raise NotFoundError(f"{kind[:-1]} '{name}' not found")
    return json.loads(f.read_text(encoding="utf-8"))


def delete_entry(kind: str, name: str):
    f = _dir_for(kind) / f"{name}.json"
    if not f.exists():
        raise NotFoundError(f"{kind[:-1]} '{name}' not found")
    f.unlink()


def _norm_star_path(path: str) -> str:
    """Canonical path for star list membership (resolved, stable across slash styles)."""
    if not path:
        return ""
    try:
        return str(Path(path).resolve())
    except Exception:  # noqa: BLE001
        return path.replace("\\", "/")


def _stars_file() -> Path:
    return workspace.root / "library" / "stars.json"


def list_stars() -> list[str]:
    f = _stars_file()
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    paths = data.get("paths") if isinstance(data, dict) else data
    if not isinstance(paths, list):
        return []
    return [_norm_star_path(p) for p in paths if isinstance(p, str) and p]


def _write_stars(paths: list[str]) -> list[str]:
    f = _stars_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"paths": paths, "updated_at": time.time()}, indent=2), encoding="utf-8")
    return paths


def toggle_star(path: str) -> list[str]:
    """Add or remove a model path from the starred list. Returns the new list."""
    if not path:
        raise ValidationError("path required")
    path = _norm_star_path(path)
    paths = list_stars()
    if path in paths:
        paths = [p for p in paths if p != path]
    else:
        paths.append(path)
    return _write_stars(paths)


def replace_star_path(old: str, new: str) -> list[str]:
    """Point stars at a renamed model file. No-op if ``old`` was not starred."""
    old = _norm_star_path(old)
    new = _norm_star_path(new)
    if not old or old == new:
        return list_stars()
    paths = list_stars()
    if old not in paths:
        return paths
    out = []
    for p in paths:
        if p == old:
            if new and new not in out:
                out.append(new)
        elif p not in out:
            out.append(p)
    return _write_stars(out)


def remove_star(path: str) -> list[str]:
    path = _norm_star_path(path)
    paths = [p for p in list_stars() if p != path]
    return _write_stars(paths)


def remove_stars_under(root: str | Path) -> list[str]:
    """Drop starred paths that live inside ``root`` (e.g. a deleted run folder)."""
    try:
        base = str(Path(root).resolve())
    except (ValueError, OSError):
        return list_stars()
    kept = []
    for p in list_stars():
        try:
            if str(Path(p).resolve()).startswith(base):
                continue
        except (ValueError, OSError):
            pass
        kept.append(p)
    return _write_stars(kept)


# --- hidden models ---------------------------------------------------
# Models Kiln finds but does not own (the install's models/ folders, anything
# linked in) cannot be deleted from Kiln. Hiding records the path here and the
# model list leaves it out; the file itself is never touched.

def _hidden_file() -> Path:
    return workspace.root / "library" / "hidden.json"


def list_hidden() -> list[str]:
    f = _hidden_file()
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    paths = data.get("paths") if isinstance(data, dict) else data
    if not isinstance(paths, list):
        return []
    return [_norm_star_path(p) for p in paths if isinstance(p, str) and p]


def _write_hidden(paths: list[str]) -> list[str]:
    f = _hidden_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"paths": paths, "updated_at": time.time()}, indent=2),
                 encoding="utf-8")
    return paths


def set_hidden(path: str, hidden: bool) -> list[str]:
    if not path:
        raise ValidationError("path required")
    path = _norm_star_path(path)
    paths = [p for p in list_hidden() if p != path]
    if hidden:
        paths.append(path)
    return _write_hidden(paths)


def replace_hidden_path(old: str, new: str) -> list[str]:
    """Keep a hidden model hidden across a rename.

    Both lists are keyed by path, so both have to follow one. Without this a
    renamed hidden model comes back into the lists and leaves a dead entry
    behind, which is worse than either outcome on its own.
    """
    old, new = _norm_star_path(old), _norm_star_path(new)
    paths = list_hidden()
    if old not in paths:
        return paths
    out = []
    for p in paths:
        if p == old:
            if new and new not in out:
                out.append(new)
        elif p not in out:
            out.append(p)
    return _write_hidden(out)


def remove_hidden_under(root: str | Path) -> list[str]:
    """Drop hidden paths inside ``root`` (e.g. a deleted run folder)."""
    try:
        base = str(Path(root).resolve())
    except (ValueError, OSError):
        return list_hidden()
    kept = []
    for p in list_hidden():
        try:
            if str(Path(p).resolve()).startswith(base):
                continue
        except (ValueError, OSError):
            pass
        kept.append(p)
    return _write_hidden(kept)


# --- model cards (display name vs training identity) -------------------

def inside_workspace(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(workspace.root.resolve())
        return True
    except (ValueError, OSError):
        return False


def owned_by_kiln(path: str | Path) -> bool:
    """Is this a file Kiln itself put in the workspace, and so Kiln's to delete?

    Inside the workspace *as written* is not enough: a link (symlink, or a
    junction on Windows) anywhere below the workspace root makes the real file
    live somewhere else -- the user's own copy -- and deleting through it would
    destroy that. So the path must be inside the workspace and resolving it must
    not leave the workspace's own real location. The install's ``models/`` and
    ``vendor/`` folders are outside the workspace and never owned.
    """
    return model_rights(path)["owned"]


def is_run_checkpoint(path: str | Path) -> bool:
    """True for trainer snapshots under ``runs/<run>/`` (including legacy project runs)."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(workspace.root.resolve())
    except (ValueError, OSError):
        return False
    return _run_parts(rel.parts)


def can_rename(path: str | Path) -> bool:
    return model_rights(path)["renamable"]


def _run_parts(parts: tuple) -> bool:
    """Do these workspace-relative parts name a trainer snapshot?"""
    if len(parts) >= 3 and parts[0] == "runs":
        return True
    # legacy: projects/<name>/runs/<run>/file.pt
    return len(parts) >= 4 and parts[0] == "projects" and parts[2] == "runs"


def model_rights(path: str | Path, root_real: str | None = None) -> dict:
    """``owned`` (``owned_by_kiln``) and ``renamable`` (``can_rename``) at once.

    The model list asks both of every model. Asked separately they resolved
    the same paths five or six times a model, and resolving a path is slow on
    Windows: it was half the list's time. Here the model's path is resolved
    once, and a caller listing many models can resolve the root once too.
    """
    rights = {"owned": False, "renamable": False}
    try:
        root_abs = Path(os.path.abspath(workspace.root))
        p = Path(os.path.abspath(path))
        root_real = root_real or os.path.realpath(root_abs)
        real = os.path.realpath(p)
    except OSError:
        return rights
    try:
        rel = p.relative_to(root_abs)
        expected = os.path.normcase(os.path.join(root_real, rel))
        rights["owned"] = os.path.normcase(real) == expected
    except ValueError:
        pass
    try:
        real_parts = Path(real).relative_to(root_real).parts
    except ValueError:
        return rights  # resolves outside the workspace: not Kiln's to rename
    rights["renamable"] = p.suffix == ".pt" and p.exists() and not _run_parts(real_parts)
    return rights


# A model's card and thumbnail live in a ".kiln" folder beside it, so a models
# folder shows the models and nothing else. They used to sit right next to the
# model; those are still read, and tidy_model_folders moves them in.
SIDECAR_DIR = ".kiln"


def _hide(d: Path):
    """A leading dot hides a folder on macOS and Linux; Windows needs the attribute."""
    if os.name != "nt":
        return
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        attrs = k32.GetFileAttributesW(str(d))
        if attrs != -1:
            k32.SetFileAttributesW(str(d), attrs | 0x2)  # FILE_ATTRIBUTE_HIDDEN
    except Exception:  # noqa: BLE001
        pass


def _sidecar_dir(folder: Path) -> Path:
    d = folder / SIDECAR_DIR
    if not d.is_dir():
        d.mkdir(parents=True, exist_ok=True)
        _hide(d)
    return d


def card_path(pt: str | Path) -> Path:
    pt = Path(pt)
    return pt.parent / SIDECAR_DIR / pt.with_suffix(".card.json").name


def _legacy_card_path(pt: str | Path) -> Path:
    return Path(pt).with_suffix(".card.json")


def thumb_path(pt: str | Path, ensure_dir: bool = False) -> Path:
    """Where a model's own thumbnail is written."""
    pt = Path(pt)
    if ensure_dir:
        _sidecar_dir(pt.parent)
    return pt.parent / SIDECAR_DIR / pt.with_suffix(".png").name


def _thumb_candidates(pt: Path) -> list[Path]:
    side = pt.parent / SIDECAR_DIR
    return [side / f"{pt.stem}.png", side / f"{pt.stem}.jpg",
            pt.with_suffix(".png"), pt.with_suffix(".jpg")]


def own_thumb(pt: str | Path) -> Path | None:
    """The model's own thumbnail, wherever it was saved; not a run's sample."""
    return next((f for f in _thumb_candidates(Path(pt)) if f.is_file()), None)


def drop_sidecars(pt: str | Path, thumbs: bool = True):
    """Remove a model's card and, unless told not to, its thumbnail."""
    files = [card_path(pt), _legacy_card_path(pt)]
    if thumbs:
        files += _thumb_candidates(Path(pt))
    for f in files:
        f.unlink(missing_ok=True)


def read_card(pt: str | Path) -> dict:
    for f in (card_path(pt), _legacy_card_path(pt)):
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def write_card(pt: str | Path, card: dict) -> dict:
    card = dict(card)
    card["updated_at"] = time.time()
    _sidecar_dir(Path(pt).parent)
    card_path(pt).write_text(json.dumps(card, indent=2), encoding="utf-8")
    _legacy_card_path(pt).unlink(missing_ok=True)
    return card


def tidy_model_folders() -> int:
    """Move cards and thumbnails that sit beside the models into ``.kiln``.

    Only Kiln's own models folders. A picture moves only when a model of the
    same name is there, so anything else someone keeps in the folder stays.
    """
    folders = [workspace.models]
    if workspace.projects.is_dir():
        folders += [p / "models" for p in workspace.projects.iterdir() if p.is_dir()]
    moved = 0
    for folder in folders:
        if not folder.is_dir():
            continue
        for f in list(folder.iterdir()):
            if not f.is_file():
                continue
            if f.name.endswith(".card.json"):
                pass
            elif f.suffix.lower() in (".png", ".jpg"):
                if not ((folder / f"{f.stem}.pt").is_file() or (folder / f.stem).is_dir()):
                    continue
            else:
                continue
            dest = _sidecar_dir(folder) / f.name
            try:
                if dest.exists():
                    f.unlink()  # a newer copy was already written there
                else:
                    f.rename(dest)
                moved += 1
            except OSError:
                pass
    return moved


def _uniq(items) -> list:
    out = []
    for x in items:
        if x and x not in out:
            out.append(x)
    return out


def ensure_card(pt: str | Path, **fields) -> dict:
    """Create or update a sidecar card. ``original_name`` is set once."""
    pt = Path(pt)
    card = read_card(pt)
    if not card:
        card = {
            "name": pt.stem,
            "original_name": fields.get("original_name") or pt.stem,
            "trained_as": [],
            "created_at": time.time(),
        }
    if not card.get("original_name"):
        card["original_name"] = fields.get("original_name") or pt.stem
    card["trained_as"] = _uniq(
        (card.get("trained_as") or [])
        + (fields.get("trained_as") or [])
        + [card.get("original_name"), pt.stem]
    )
    for k, v in fields.items():
        if k in ("original_name", "trained_as"):
            continue
        if v is not None:
            card[k] = v
    card.setdefault("name", pt.stem)
    _stamp_fingerprint(pt, card)
    return write_card(pt, card)


# --- lineage ----------------------------------------------------------------
# Where a model's weights came from: a lineage on its card names its parents,
# each by path and by fingerprint. The path is how it was found; the fingerprint
# (SHA-256 of the weights file) is how it is found again after a rename or a
# copy. The Merge tab uses it to default a base; a lineage view will draw it.
#
# kind      how the weights came to be: scratch, continue, finetune, lora,
#           merge, convert, copy, import
# parents   [{role, path, name, fingerprint, size, step}], role being start
#           (continue), base (finetune, lora, a merge's base), a / b (merge),
#           or source (convert, copy)


def weights_file(path: str | Path) -> Path | None:
    """The file a model's weights live in: the .pt itself, or a Diffusers repo's UNet."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"):
            for f in (p / name, p / "unet" / name):
                if f.is_file():
                    return f
    return None


def _stamp(f: Path) -> list:
    st = f.stat()
    return [st.st_size, st.st_mtime_ns]


def fingerprint(path: str | Path, card: dict | None = None) -> str | None:
    """SHA-256 of a model's weights, from its card while the file is unchanged."""
    f = weights_file(path)
    if f is None:
        return None
    card = read_card(path) if card is None else card
    if card.get("fingerprint") and card.get("fingerprint_stamp") == _stamp(f):
        return card["fingerprint"]
    from app.core.sample_models import sha256_of

    return sha256_of(f)


def _stamp_fingerprint(pt: Path, card: dict):
    """Keep a card's own fingerprint current; hashing only when the file changed."""
    f = weights_file(pt)
    if f is None:
        return
    stamp = _stamp(f)
    if card.get("fingerprint_stamp") != stamp or not card.get("fingerprint"):
        card["fingerprint"] = fingerprint(pt, card={})
        card["fingerprint_stamp"] = stamp


def parent_ref(path: str | Path, role: str, step: int | None = None) -> dict:
    """One parent as a lineage records it.

    ``path`` may name something that is not a file -- a Diffusers base can be a
    Hub id -- in which case it is recorded by name only, with no fingerprint.
    """
    p = Path(str(path))
    card = read_card(p) if p.exists() else {}
    f = weights_file(p)
    return {
        "role": role,
        "path": str(path),
        "name": card.get("name") or (p.name if p.is_dir() else p.stem),
        "fingerprint": fingerprint(p, card) if f is not None else None,
        "size": f.stat().st_size if f is not None else None,
        "step": step if step is not None else card.get("step"),
    }


def lineage(kind: str, parents: list | None = None, **extra) -> dict:
    return {"kind": kind, "parents": list(parents or []),
            **{k: v for k, v in extra.items() if v is not None}}


def run_lineage(run_dir: str | Path) -> dict | None:
    """What a training run started from, read from its run.json.

    xurdif records the file it loaded as ``resume``; a Diffusers run records
    its ``mode`` and ``base_model``. None when the folder is not a run.
    """
    from app.core.engine.trainer import run_meta

    run_dir = Path(run_dir)
    if not (run_dir / "run.json").is_file():
        return None
    meta = run_meta(run_dir)
    run = meta.get("name") or run_dir.name
    if meta.get("backend") == "diffusers":
        mode = meta.get("mode") or "scratch"
        base = meta.get("base_model")
        if mode in ("finetune", "lora") and base:
            return lineage(mode, [parent_ref(base, "base")], run=run)
        return lineage("scratch", run=run)
    start = meta.get("resume")
    if start:
        return lineage("continue", [parent_ref(start, "start")], run=run)
    return lineage("scratch", run=run)


def lineage_of(path: str | Path) -> dict:
    """A model's lineage: from its card, else from the run it is a snapshot of."""
    recorded = read_card(path).get("lineage")
    if recorded:
        return recorded
    return run_lineage(Path(path).parent) or lineage("unknown")


def _library_model_paths() -> list[Path]:
    folders = [workspace.models]
    if workspace.projects.is_dir():
        folders += [p / "models" for p in workspace.projects.iterdir() if p.is_dir()]
    out = []
    for folder in folders:
        if folder.is_dir():
            out += [f for f in folder.iterdir()
                    if not f.name.startswith(".") and (f.suffix == ".pt" or f.is_dir())]
    return out


def resolve_parent(parent: dict, candidates: list | None = None) -> str | None:
    """Where a recorded parent is now: its path, or a file with its fingerprint.

    Candidates default to Kiln's model folders. A candidate's card fingerprint
    is trusted while its file is unchanged; otherwise only files of the
    parent's size are hashed, so a search reads few files.
    """
    fp = parent.get("fingerprint")
    p = parent.get("path")
    if p and weights_file(p) is not None and (not fp or fingerprint(p) == fp):
        return str(p)
    if not fp:
        return None
    for c in (_library_model_paths() if candidates is None else candidates):
        f = weights_file(c)
        if f is None:
            continue
        card = read_card(c)
        if card.get("fingerprint_stamp") == _stamp(f):
            if card.get("fingerprint") == fp:
                return str(c)
            continue
        if parent.get("size") in (None, f.stat().st_size) and fingerprint(c, {}) == fp:
            return str(c)
    return None


def sibling_thumb(src: str | Path) -> Path | None:
    src = Path(src)
    own = own_thumb(src)
    if own is not None:
        return own
    if src.stem.startswith("model-"):
        try:
            sample = src.with_name(f"sample-{src.stem.split('-', 1)[1]}.png")
        except IndexError:
            sample = None
        if sample is not None and sample.exists():
            return sample
    return None


def delete_model_files(path: str | Path):
    """Remove a workspace ``.pt`` plus its thumbnail and card sidecars."""
    p = Path(path)
    if not owned_by_kiln(p):
        raise ValidationError(
            "this model is not in Kiln's workspace (or is linked in from elsewhere), "
            "so Kiln will not delete it; hide it instead")
    if not p.exists():
        raise NotFoundError("model not found")
    p.unlink()
    # The model's own thumbnail goes with it; a run's sample-N.png stays.
    drop_sidecars(p, thumbs=not is_run_checkpoint(p))
    remove_star(str(p))


def rename_model(path: str, new_name: str) -> dict:
    """Rename a library/project model file. Keeps ``original_name`` on the card."""
    src = Path(path)
    if not src.exists() or src.suffix != ".pt":
        raise NotFoundError("model not found")
    if not inside_workspace(src):
        raise ValidationError("refusing to rename a file outside the workspace")
    if is_run_checkpoint(src):
        raise ValidationError(
            "run snapshots keep their trainer filenames — save to the library first, then rename"
        )
    new_name = safe_name(new_name, "model name")
    dest = src.with_name(f"{new_name}.pt")
    if dest.resolve() == src.resolve():
        card = ensure_card(src, name=new_name)
        return {"path": str(src), "name": new_name, "original_name": card.get("original_name"), "card": card}

    if dest.exists():
        raise ValidationError(f"a model named '{new_name}' already exists")

    old_stem = src.stem
    card = read_card(src)
    src.rename(dest)
    thumb = own_thumb(src)
    if thumb is not None:
        thumb.replace(thumb_path(dest, ensure_dir=True).with_suffix(thumb.suffix))
    drop_sidecars(src)

    if not card.get("original_name"):
        card["original_name"] = old_stem
    card["name"] = new_name
    card["trained_as"] = _uniq((card.get("trained_as") or []) + [old_stem, card.get("original_name"), new_name])
    write_card(dest, card)
    replace_star_path(str(src), str(dest))
    replace_hidden_path(str(src), str(dest))
    return {
        "path": str(dest),
        "name": new_name,
        "original_name": card.get("original_name"),
        "trained_as": card.get("trained_as") or [],
        "card": card,
    }
