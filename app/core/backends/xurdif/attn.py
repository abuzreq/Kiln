"""The attention layout of a ``tinyunet_conf_attention`` model.

Upstream spells the layout as ``LOCATION:TYPE`` pairs -- ``"-1:linear,mid:full"``
-- where ``mid`` is the bottleneck and ``-k`` is the k-th encoder level above it,
and the type is ``full``, ``linear`` or ``window``. The parser here mirrors
``alt_models.tinyunet_conf_attn.parse_attn_config`` exactly (a smoke test holds
them to it) but stays pure Python so validating a training request does not
import torch, and adds the one thing upstream lacks: a canonical spelling, so
two specs that mean the same layout compare equal as strings.

Kiln always stores the canonical string. ``None`` (upstream's "use the
constructor default") is never written by Kiln; it survives only in
``describe()`` for checkpoints trained elsewhere without the metadata.
"""
from utils.exceptions import ValidationError

MTYPE = "tinyunet_conf_attention"
KINDS = ("full", "linear", "window")
NONE = "none"

# The vendor's own example: linear attention one level above the bottleneck is
# cheap (no N x N matrix) and buys some composition; full attention there would
# be N^2 at four times the tokens and would not fit the presets at 512px.
DEFAULT_SPEC = "-1:linear,mid:full"

# What the Train screen offers by name. ``spec`` is canonical.
NAMED_LAYOUTS = [
    {"id": "mid-linear", "label": "Bottleneck + linear above", "spec": DEFAULT_SPEC,
     "tip": "Full attention at the bottleneck and cheap linear attention one level up. "
            "The vendor's suggestion, and the default for new models."},
    {"id": "mid", "label": "Bottleneck only", "spec": "mid:full",
     "tip": "One full-attention layer at the bottleneck, the shape every earlier Kiln "
            "model had. Cheapest layout with attention."},
    {"id": "mid-window", "label": "Bottleneck + window above", "spec": "-1:window,mid:full",
     "tip": "Full attention at the bottleneck and attention inside 8x8 windows one level "
            "up. Local detail rather than long-range composition."},
    {"id": "none", "label": "No attention", "spec": NONE,
     "tip": "A plain convolutional network. Fastest, and weakest at overall composition."},
]


def parse(spec) -> dict | None:
    """Spec string -> ``{-1: "linear", "mid": "full"}``; ``None`` when unspecified.

    Same grammar and the same rejections as upstream's parser, phrased for the
    person who typed the value rather than for a traceback.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        return dict(spec)
    value = str(spec).strip()
    if not value:
        return None
    if value.lower() == NONE:
        return {}
    config: dict = {}
    for item in value.split(","):
        item = item.strip()
        if ":" not in item:
            raise ValidationError(
                f"attention layout '{item}' should read LOCATION:TYPE, e.g. -1:linear or mid:full")
        location, kind = item.split(":", 1)
        location, kind = location.strip(), kind.strip().lower()
        if kind not in KINDS:
            raise ValidationError(
                f"unknown attention type '{kind}'; use one of {', '.join(KINDS)}")
        if location.lower() == "mid":
            key: object = "mid"
        else:
            try:
                key = int(location)
            except ValueError:
                raise ValidationError(
                    f"attention location '{location}' should be 'mid' or a negative level "
                    "such as -1 (the level just above the bottleneck)")
            if key >= 0:
                raise ValidationError(
                    f"attention level {key} is not negative; -1 is the level just above "
                    "the bottleneck, -2 the one above that")
        if key in config:
            raise ValidationError(f"attention location '{location}' is given twice")
        config[key] = kind
    return config


def canonical(spec) -> str | None:
    """Deepest encoder level first, ``mid`` last; ``{}`` spells ``none``."""
    config = parse(spec)
    if config is None:
        return None
    if not config:
        return NONE
    levels = sorted(k for k in config if k != "mid")
    parts = [f"{k}:{config[k]}" for k in levels]
    if "mid" in config:
        parts.append(f"mid:{config['mid']}")
    return ",".join(parts)


def validate(spec, mults) -> str:
    """Canonical spec, or a ``ValidationError`` that names the problem now.

    A location deeper than the network fails inside the constructor several
    seconds into the training subprocess otherwise, the way a bad ``mults``
    once did (see ``_clean_mults``).
    """
    config = parse(spec)
    if config is None:
        config = parse(DEFAULT_SPEC)
    n_levels = len(list(mults or []))
    for k in config:
        if k != "mid" and -int(k) > n_levels:
            raise ValidationError(
                f"attention level {k} is outside the network: with {n_levels} channel "
                f"multipliers the levels run from -1 to -{n_levels}")
    return canonical(config)


def layout_name(spec) -> str | None:
    """The named layout a spec matches, if any (for labels and the UI)."""
    c = canonical(spec)
    for layout in NAMED_LAYOUTS:
        if layout["spec"] == c:
            return layout["id"]
    return None
