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


def set_hidden(path: str, hidden: bool) -> list[str]:
    if not path:
        raise ValidationError("path required")
    path = _norm_star_path(path)
    paths = [p for p in list_hidden() if p != path]
    if hidden:
        paths.append(path)
    f = _hidden_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"paths": paths, "updated_at": time.time()}, indent=2), encoding="utf-8")
    return paths


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
    try:
        root = Path(os.path.abspath(workspace.root))
        p = Path(os.path.abspath(path))
        rel = p.relative_to(root)
    except (ValueError, OSError):
        return False
    try:
        real = os.path.normcase(os.path.realpath(p))
        expected = os.path.normcase(os.path.join(os.path.realpath(root), rel))
    except OSError:
        return False
    return real == expected


def is_run_checkpoint(path: str | Path) -> bool:
    """True for trainer snapshots under ``runs/<run>/`` (including legacy project runs)."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(workspace.root.resolve())
    except (ValueError, OSError):
        return False
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "runs":
        return True
    # legacy: projects/<name>/runs/<run>/file.pt
    return len(parts) >= 4 and parts[0] == "projects" and parts[2] == "runs"


def can_rename(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.suffix == ".pt" and inside_workspace(p) and not is_run_checkpoint(p)


def card_path(pt: str | Path) -> Path:
    return Path(pt).with_suffix(".card.json")


def read_card(pt: str | Path) -> dict:
    f = card_path(pt)
    if not f.exists():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def write_card(pt: str | Path, card: dict) -> dict:
    card = dict(card)
    card["updated_at"] = time.time()
    card_path(pt).write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


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
    return write_card(pt, card)


def sibling_thumb(src: str | Path) -> Path | None:
    src = Path(src)
    png = src.with_suffix(".png")
    if png.exists():
        return png
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
    thumb = sibling_thumb(p)
    card = card_path(p)
    p.unlink()
    if thumb is not None and thumb.exists() and thumb != p:
        # only delete a true sidecar sitting next to the .pt (not a run sample
        # we might still want if this were a checkpoint — but we refuse those)
        if thumb.suffix.lower() == ".png" and thumb.parent == p.parent:
            if not is_run_checkpoint(p):
                thumb.unlink(missing_ok=True)
    card.unlink(missing_ok=True)
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
    src.rename(dest)
    png = src.with_suffix(".png")
    if png.exists():
        png.rename(dest.with_suffix(".png"))
    old_card = card_path(src)
    if old_card.exists():
        old_card.rename(card_path(dest))

    card = read_card(dest)
    if not card.get("original_name"):
        card["original_name"] = old_stem
    card["name"] = new_name
    card["trained_as"] = _uniq((card.get("trained_as") or []) + [old_stem, card.get("original_name"), new_name])
    write_card(dest, card)
    replace_star_path(str(src), str(dest))
    return {
        "path": str(dest),
        "name": new_name,
        "original_name": card.get("original_name"),
        "trained_as": card.get("trained_as") or [],
        "card": card,
    }
