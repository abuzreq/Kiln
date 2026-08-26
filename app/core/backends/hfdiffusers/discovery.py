"""Deciding whether a model is usable before committing to downloading it.

The rule this module exists to enforce: never pull gigabytes of weights to find
out a repo is unsupported. Everything here reads small JSON metadata only, and
every refusal carries a reason written for a person to read.

What a user may paste is deliberately broad -- a repo id, a full Hub URL, an
explicit ``diffusers:`` ref, or a local folder -- because the alternative is a
model browser that rejects the thing the user just copied out of their address
bar. What gets *loaded* stays narrow.
"""
import re
from pathlib import Path

from utils.exceptions import ValidationError
from utils.logger import get_logger

from . import loader

log = get_logger("diffusers.discovery")

_HUB_URL = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/(?P<repo>[^/\s]+/[^/\s?#]+)"
    r"(?:/tree/(?P<rev>[^/\s?#]+))?/?.*$", re.I)
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

# Repo ids we know are worth offering. Small, unconditional, pixel-space, and
# each one verified to satisfy probe(). Everything else the user pastes still
# goes through validate() -- this list is a starting point, not a whitelist.
SUGGESTED = [
    {"repo": "google/ddpm-celebahq-256", "label": "DDPM CelebA-HQ 256",
     "note": "Faces. The canonical unconditional DDPM."},
    {"repo": "google/ddpm-ema-celebahq-256", "label": "DDPM CelebA-HQ 256 (EMA)",
     "note": "Averaged weights; usually the smoother of the two."},
    {"repo": "google/ddpm-church-256", "label": "DDPM LSUN Church 256",
     "note": "Buildings and skies. Good raw material for bending."},
    {"repo": "google/ddpm-bedroom-256", "label": "DDPM LSUN Bedroom 256",
     "note": "Interiors."},
    {"repo": "google/ddpm-cat-256", "label": "DDPM Cats 256",
     "note": "Cats."},
    {"repo": "nvidia/CIFAR-10-DDPM", "label": "DDPM CIFAR-10 32",
     "note": "Tiny and quick; useful for trying the pipeline out."},
]


def normalize(text: str) -> tuple[str, str | None]:
    """Turn whatever the user pasted into ``(locator, revision)``.

    Raises ValidationError rather than guessing when the string is not a shape
    we recognise: a silent wrong guess here becomes a confusing 404 later.
    """
    s = (text or "").strip()
    if not s:
        raise ValidationError("no model given")

    m = _HUB_URL.match(s)
    if m:
        return m.group("repo"), m.group("rev")

    for prefix in ("diffusers:", "hf:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break

    locator, _, rev = s.partition("@")
    locator = locator.strip()
    revision = rev.strip() or None

    p = Path(locator)
    try:
        if p.exists():
            return str(p), revision
    except OSError:
        pass

    if not _REPO_ID.match(locator):
        raise ValidationError(
            f"'{text}' is not a local folder, a Hub repo id (owner/name), "
            "or a huggingface.co URL")
    return locator, revision


def validate(text: str) -> dict:
    """Can Kiln use this model, and if not, exactly why?

    Metadata only -- no weights are fetched. The result is shaped for direct
    display: ``ok`` drives the button, ``reason`` fills the error line, and
    ``requires`` warns about anything the user would otherwise discover halfway
    through a download.
    """
    from app.core import backends

    try:
        locator, revision = normalize(text)
    except ValidationError as e:
        return {"ok": False, "input": text, "reason": str(e)}

    is_local = Path(locator).exists()
    out: dict = {
        "ok": False,
        "input": text,
        "locator": locator,
        "revision": revision,
        "local": is_local,
        "ref": f"diffusers:{locator}" + (f"@{revision}" if revision else ""),
        "offline": loader.offline(),
    }

    try:
        info = loader.probe(locator, revision)
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"could not read model metadata: {e}"
        return out

    if not info.get("supported"):
        out["reason"] = info.get("reason") or "unsupported model"
        out["pipeline"] = info.get("pipeline")
        if not is_local and out["offline"]:
            out["reason"] += (" (offline mode is on, so this may just mean the "
                              "model is not in the local cache)")
        return out

    try:
        meta = loader.describe(locator, revision)
    except Exception as e:  # noqa: BLE001
        out["reason"] = str(e)
        return out

    caps = backends.get("diffusers").capabilities
    out.update({
        "ok": True,
        "reason": "",
        "pipeline": info.get("pipeline"),
        "model": meta.to_dict(),
        "capabilities": caps.to_dict(),
        "requires": _requirements(meta),
        "operations": _operations(caps),
    })
    return out


def _requirements(meta) -> list[str]:
    """Anything the user should know before committing to this model."""
    out = []
    size = meta.extra.get("sample_size")
    if size:
        out.append(f"trained at {size}x{size}; sampling far from that size degrades output")
    step = meta.size_multiple
    out.append(f"image dimensions must be multiples of {step}")
    for pkg, why in (("accelerate", "faster, lower-memory loading and multi-GPU training"),
                     ("peft", "LoRA fine-tuning")):
        if not _installed(pkg):
            out.append(f"'{pkg}' is not installed ({why})")
    return out


def _operations(caps) -> dict:
    return {"sample": caps.inference, "bend": caps.bend, "merge": caps.merge,
            "finetune": caps.finetune, "lora": caps.lora,
            "train_from_scratch": caps.train_from_scratch}


def _installed(pkg: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(pkg) is not None


def suggested() -> list[dict]:
    """The starter list, annotated with whether each is already cached."""
    out = []
    for entry in SUGGESTED:
        item = dict(entry)
        item["ref"] = f"diffusers:{entry['repo']}"
        item["cached"] = _is_cached(entry["repo"])
        out.append(item)
    return out


def _is_cached(repo: str) -> bool:
    """Is this repo's metadata already local? Never touches the network."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return False
    for name in ("model_index.json", "config.json"):
        try:
            hit = try_to_load_from_cache(repo, name)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(hit, str):
            return True
    return False


def import_to_workspace(text: str, name: str, dest_root, progress=None) -> dict:
    """Download a validated model into the workspace so it lists like any other.

    Validation runs first and hard-stops on failure, so an unsupported repo
    costs one metadata read rather than a full download.
    """
    from huggingface_hub import snapshot_download

    from utils.validators import safe_name

    verdict = validate(text)
    if not verdict["ok"]:
        raise ValidationError(verdict.get("reason") or "unsupported model")

    name = safe_name(name or verdict["model"]["name"], "model name")
    dest = Path(dest_root) / name
    if dest.exists() and any(dest.iterdir()):
        raise ValidationError(f"a model folder named '{name}' already exists")

    if verdict["local"]:
        import shutil

        shutil.copytree(verdict["locator"], dest, dirs_exist_ok=True)
    else:
        if progress:
            progress(f"downloading {verdict['locator']}...")
        snapshot_download(
            verdict["locator"], revision=verdict["revision"], local_dir=str(dest),
            # Weights plus the small configs; skip the duplicated .bin when a
            # safetensors copy exists, and never pull the training extras.
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"],
            ignore_patterns=["*.msgpack", "*.onnx", "*.onnx_data", "*.ckpt"],
        )

    loader.forget(str(dest))
    if not loader.is_local_model_dir(dest):
        raise ValidationError(
            "the downloaded folder does not look like a Diffusers model")
    meta = loader.describe(str(dest))
    log.info("imported %s -> %s", verdict["locator"], dest)
    return {"path": str(dest), "name": name, "model": meta.to_dict(),
            "source": verdict["locator"]}
