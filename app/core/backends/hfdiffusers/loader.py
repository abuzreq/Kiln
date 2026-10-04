"""Locating and loading Diffusers models.

Only unconditional, pixel-space ``UNet2DModel`` is supported. That is not a
placeholder for "more later": it is the family whose forward pass has the same
shape as xurdif's, so Kiln's existing denoise loop, img2img, mask compositing
and bending all apply to it unchanged. Latent and text-conditioned pipelines
need an encode/decode step and a conditioning argument that the loop has no way
to express, so they are refused by name rather than half-supported.

Two repository layouts exist in the wild and both are handled:

- flat/legacy -- ``config.json`` at the root is the UNet itself
  (google/ddpm-*, saved by very old diffusers)
- pipeline -- ``model_index.json`` at the root, UNet under ``unet/``
  (anything saved by ``DiffusionPipeline.save_pretrained``)
"""
import json
import os
import threading
from pathlib import Path

from app.core.backends.base import ModelDescriptor
from utils.exceptions import IncompatibleModelError, NotFoundError
from utils.logger import get_logger

log = get_logger("diffusers.loader")

# Model classes this backend loads. Both are unconditional and pixel-space --
# that is the requirement, not the class name. TinyUNet2DModel is xurdif's
# compact architecture re-homed (see tinyunet.py); it is far cheaper to run than
# any UNet2DModel config, so it is a first-class option, not a legacy shim.
SUPPORTED_MODEL_CLASS = "UNet2DModel"
TINYUNET_CLASS = "TinyUNet2DModel"
SUPPORTED_MODEL_CLASSES = (SUPPORTED_MODEL_CLASS, TINYUNET_CLASS)
# Pipelines whose `unet` entry is a plain UNet2DModel we can drive directly.
SUPPORTED_PIPELINES = {"DDPMPipeline", "DDIMPipeline", "PNDMPipeline"}
# Where the UNet config may live, in probe order.
UNET_SUBFOLDERS = (None, "unet")
SCHEDULER_SUBFOLDERS = (None, "scheduler")

# diffusers prediction_type -> the vocabulary Kiln's denoise loop speaks.
PREDICTION_TYPES = {"epsilon": "eps", "sample": "x0"}

_describe_cache: dict = {}
_lock = threading.Lock()


def offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip() not in ("", "0", "false", "False")


def is_local_model_dir(path: Path) -> bool:
    """Cheap structural check -- no torch, no network, no weight reads."""
    try:
        if not path.is_dir():
            return False
    except OSError:
        return False
    if (path / "model_index.json").is_file():
        return True
    cfg = path / "config.json"
    if cfg.is_file():
        try:
            return json.loads(cfg.read_text(encoding="utf-8")).get("_class_name") is not None
        except Exception:  # noqa: BLE001
            return False
    return (path / "unet" / "config.json").is_file()


def _read_json(locator: str, filename: str, subfolder: str | None = None,
               revision: str | None = None) -> dict | None:
    """Read one small JSON from a local directory or the Hub cache.

    Only ever fetches metadata files, never weights: the point of the discovery
    step is to decide whether a repo is usable *before* pulling gigabytes of
    tensors for something we would refuse anyway.
    """
    p = Path(locator)
    try:
        is_dir = p.is_dir()
    except OSError:
        is_dir = False
    if is_dir:
        f = (p / subfolder / filename) if subfolder else (p / filename)
        if not f.is_file():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("unreadable %s: %s", f, e)
            return None

    from huggingface_hub import hf_hub_download

    try:
        got = hf_hub_download(locator, filename, subfolder=subfolder,
                              revision=revision, local_files_only=offline())
    except Exception as e:  # noqa: BLE001
        # Missing file, offline with a cold cache, gated repo, no network -- all
        # "cannot answer", and the caller turns that into a readable message.
        log.info("could not fetch %s from %s: %s", filename, locator, e)
        return None
    try:
        return json.loads(Path(got).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("unreadable %s from %s: %s", filename, locator, e)
        return None


def probe(locator: str, revision: str | None = None) -> dict:
    """Decide whether a repo/dir is loadable, without loading it.

    Returns ``{supported, reason, pipeline, unet_subfolder, scheduler_subfolder,
    unet_config, scheduler_config}``. ``reason`` is written to be shown to a
    user verbatim, because "unsupported" with no explanation is the single most
    frustrating thing a model browser can say.
    """
    index = _read_json(locator, "model_index.json", revision=revision)
    pipeline = (index or {}).get("_class_name")

    unet_cfg = None
    unet_sub = None
    root_class = None
    for sub in UNET_SUBFOLDERS:
        cfg = _read_json(locator, "config.json", subfolder=sub, revision=revision)
        if cfg is None:
            continue
        if cfg.get("_class_name") in SUPPORTED_MODEL_CLASSES:
            unet_cfg, unet_sub = cfg, sub
            break
        if sub is None:
            root_class = cfg.get("_class_name")

    if unet_cfg is None:
        if pipeline and pipeline not in SUPPORTED_PIPELINES:
            return {"supported": False, "pipeline": pipeline,
                    "reason": f"'{pipeline}' is a conditional or latent pipeline; this "
                              f"backend only drives unconditional pixel-space "
                              f"{SUPPORTED_MODEL_CLASS} models"}
        if root_class:
            return {"supported": False, "pipeline": pipeline,
                    "reason": f"unsupported model class '{root_class}'; this "
                              f"backend loads "
                              f"{' and '.join(SUPPORTED_MODEL_CLASSES)}"}
        return {"supported": False, "pipeline": pipeline,
                "reason": f"no {SUPPORTED_MODEL_CLASS} config found (looked for "
                          f"config.json and unet/config.json)"}

    if pipeline and pipeline not in SUPPORTED_PIPELINES:
        return {"supported": False, "pipeline": pipeline,
                "reason": f"'{pipeline}' is not a supported pipeline"}

    sched_cfg = None
    sched_sub = None
    for sub in SCHEDULER_SUBFOLDERS:
        cfg = _read_json(locator, "scheduler_config.json", subfolder=sub, revision=revision)
        if cfg:
            sched_cfg, sched_sub = cfg, sub
            break

    pred_raw = (sched_cfg or {}).get("prediction_type", "epsilon")
    if pred_raw not in PREDICTION_TYPES:
        # v_prediction would silently corrupt the eps/x0 conversion in the loop.
        return {"supported": False, "pipeline": pipeline,
                "reason": f"prediction_type '{pred_raw}' is not supported "
                          f"(need one of {sorted(PREDICTION_TYPES)})"}

    return {
        "supported": True,
        "reason": "",
        "pipeline": pipeline,
        "unet_subfolder": unet_sub,
        "scheduler_subfolder": sched_sub,
        "unet_config": unet_cfg,
        "scheduler_config": sched_cfg or {},
    }


def shape_of(cfg: dict) -> tuple[list, int]:
    """``(mults, size_multiple)`` for one model config.

    The two architectures describe their widths differently -- UNet2DModel by
    absolute ``block_out_channels``, TinyUNet2DModel by multipliers over a base
    ``dim``, which is xurdif's own vocabulary -- and they downsample a different
    number of times for the same number of levels.
    """
    if cfg.get("_class_name") == TINYUNET_CLASS:
        mults = [int(m) for m in (cfg.get("dim_mults") or [1])]
        # one downsample per level
        return mults, 2 ** len(mults)
    channels = list(cfg.get("block_out_channels") or [])
    return mults_from(channels), size_multiple(channels)


def mults_from(block_out_channels: list) -> list:
    """Express block widths the way Kiln already talks about them.

    xurdif's ``mults`` are channel multipliers over a base width, which is the
    same information ``block_out_channels`` carries absolutely. Converting means
    the existing UI chips and model subtitles read correctly for both engines.
    """
    if not block_out_channels:
        return [1]
    base = block_out_channels[0]
    return [int(c // base) if base else 1 for c in block_out_channels]


def size_multiple(block_out_channels: list) -> int:
    """UNet2DModel downsamples at every down block but the last."""
    return 2 ** max(len(block_out_channels or [1]) - 1, 0)


def _local_size_mb(locator: str) -> float:
    p = Path(locator)
    try:
        if not p.is_dir():
            return 0.0
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
    except OSError:
        return 0.0


def describe(locator: str, revision: str | None = None) -> ModelDescriptor:
    key = (locator, revision)
    with _lock:
        hit = _describe_cache.get(key)
    if hit is not None:
        return hit.copy()

    info = probe(locator, revision)
    if not info["supported"]:
        raise IncompatibleModelError(f"{locator}: {info['reason']}")

    # Kiln's own notes about a model it produced (a conversion, so far). Absent
    # for anything from the Hub, and namespaced so it can never collide with
    # the model's real config.
    provenance = _read_json(locator, "kiln_provenance.json", revision=revision) or {}

    cfg = info["unet_config"]
    channels = list(cfg.get("block_out_channels") or [])
    cls_name = cfg.get("_class_name", SUPPORTED_MODEL_CLASS)
    mults, step_mult = shape_of(cfg)
    sched = info.get("scheduler_config") or {}
    name = Path(locator).name if Path(locator).is_dir() else locator.split("/")[-1]

    meta = ModelDescriptor(
        path=locator,
        name=name,
        mtype=f"diffusers:{cls_name}",
        mults=mults,
        pred=PREDICTION_TYPES.get(sched.get("prediction_type", "epsilon"), "eps"),
        step=provenance.get("source_step"),
        size_mb=_local_size_mb(locator),
        source="workspace",
        backend="diffusers",
        size_multiple=step_mult,
        # A Diffusers repo publishes one set of weights. Repos that shipped an
        # EMA variant did so as a separate repo (google/ddpm-ema-* alongside
        # google/ddpm-*), and a converted xurdif model has already had one slot
        # chosen for it at conversion time -- recorded under extra["provenance"].
        ema="none",
        sample_size=cfg.get("sample_size"),
        extra={
            "pipeline": info.get("pipeline"),
            "model_class": cls_name,
            "unet_subfolder": info.get("unet_subfolder"),
            "scheduler_subfolder": info.get("scheduler_subfolder"),
            "sample_size": cfg.get("sample_size"),
            "block_out_channels": channels,
            "num_train_timesteps": int(sched.get("num_train_timesteps", 1000)),
            "beta_schedule": sched.get("beta_schedule", "linear"),
            "beta_start": float(sched.get("beta_start", 0.0001)),
            "beta_end": float(sched.get("beta_end", 0.02)),
            "trained_betas": sched.get("trained_betas"),
            "display": provenance.get("display"),
            "provenance": provenance or None,
        },
    )
    with _lock:
        _describe_cache[key] = meta.copy()
    return meta


def forget(locator: str | None = None):
    """Drop cached probe results (a re-download or an edited config)."""
    with _lock:
        if locator is None:
            _describe_cache.clear()
        else:
            for k in [k for k in _describe_cache if k[0] == locator]:
                del _describe_cache[k]


class UNet2DAdapter:
    """Present a UNet2DModel through xurdif's ``(x, t) -> Tensor`` contract.

    Everything downstream -- the denoise loop, the bending hooks, the layer
    probe -- was written against that signature. ``UNet2DModel.forward`` takes
    ``(sample, timestep, class_labels=None, return_dict=True)`` and returns a
    ``UNet2DOutput``, so the whole adaptation is unwrapping ``.sample``.

    Deliberately not an ``nn.Module`` subclass: wrapping would insert a level
    into every ``named_modules()`` path, so a bend saved against ``mid_block``
    would stop resolving. ``__getattr__`` forwards instead, leaving the module
    tree the rest of the app walks as the real one.
    """

    def __init__(self, net):
        object.__setattr__(self, "_net", net)

    def __call__(self, x, t):
        return self._net(x, t, return_dict=False)[0]

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_net"), item)

    def __repr__(self):
        return f"UNet2DAdapter({type(self._net).__name__})"

    @property
    def wrapped(self):
        return object.__getattribute__(self, "_net")


def load_net(locator: str, revision: str | None = None, device: str = "cpu",
             ema: bool = True):
    """Load a UNet2DModel and return ``(adapter, descriptor)``.

    ``ema`` is accepted and ignored: a Diffusers repo publishes one set of
    weights, and repos that shipped an EMA variant did so as a separate repo
    (google/ddpm-ema-* vs google/ddpm-*). Accepting the flag keeps the model
    cache key and every call site identical across backends.
    """
    from app.core.engine import warmup

    warmup.wait_ready()  # diffusers' lazy imports are not safe beside the launch warm-up
    from diffusers import UNet2DModel

    from .tinyunet import TinyUNet2DModel

    meta = describe(locator, revision)
    sub = meta.extra.get("unet_subfolder")
    cls = TinyUNet2DModel if meta.extra.get("model_class") == TINYUNET_CLASS else UNet2DModel
    try:
        net = cls.from_pretrained(
            locator, subfolder=sub, revision=revision, local_files_only=offline(),
        )
    except Exception as e:  # noqa: BLE001
        raise NotFoundError(f"could not load {locator}: {e}") from e

    if cls is TinyUNet2DModel:
        _detach_from_mmap(net)
    net = net.eval().to(device)
    return UNet2DAdapter(net), meta


def _detach_from_mmap(net):
    """Copy weights out of safetensors' memory-mapped buffer.

    Loading through safetensors leaves parameters backed by an mmap, and that
    makes the convolution kernels pick a different vectorised path: measured,
    the same weights give outputs differing by ~6e-7, which survives the
    sampling loop as a 1-in-255 difference on about one pixel in twelve
    thousand. Harmless in isolation -- but a converted xurdif model must
    reproduce its original images exactly, or every recipe card a user has
    saved quietly stops being reproducible.

    Only done for TinyUNet2DModel: these are a few MB (~12 ms, ~14 MB for the
    default shape), and they are the only models with a pre-existing library of
    outputs to stay faithful to. A large UNet2DModel has no such reference and
    would pay hundreds of MB for nothing.
    """
    for p in net.parameters():
        p.data = p.data.clone()
    for b in net.buffers():
        b.data = b.data.clone()
    return net


def scan(sources) -> list[ModelDescriptor]:
    """Model directories sitting in the folders Kiln already looks in."""
    return list(iter_scan(sources))


def iter_scan(sources):
    """``scan``, yielding each model directory as soon as it has been read."""
    seen: set[str] = set()
    for d, label in sources:
        if not d.exists():
            continue
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            continue
        for child in children:
            if str(child) in seen or not is_local_model_dir(child):
                continue
            seen.add(str(child))
            try:
                meta = describe(str(child))
            except Exception as e:  # noqa: BLE001
                log.info("skipping %s: %s", child, e)
                continue
            meta.source = label
            thumb = child / "thumbnail.png"
            if thumb.exists():
                meta.thumbnail = str(thumb)
            yield meta
