"""Smoke test for the configurable-attention architecture (CPU, no GPU needed).

Holds Kiln's pure-Python spec parser to the vendored one, builds every named
layout across network depths, round-trips the layout through a checkpoint in
each format that records it, and checks that a conf checkpoint with an
unrecorded or wrong layout is refused rather than silently loaded with missing
keys.

    python scripts/smoke_conf_attention.py
"""
import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app.core.backends.xurdif import attn, loader  # noqa: E402
from app.core.engine._vendor import ensure_on_path  # noqa: E402
from app.core.engine.arch import CONF_MTYPE, build_unet  # noqa: E402
from utils.exceptions import ValidationError  # noqa: E402

ensure_on_path()
from alt_models.tinyunet_conf_attn import parse_attn_config  # noqa: E402

README_EXAMPLES = ["mid:full", "-1:linear,mid:full", "-1:linear,mid:linear",
                   "-2:window,-1:linear,mid:full", "none"]


def check_parser_parity():
    specs = README_EXAMPLES + [None, "", "  ", "MID:FULL", "-1 : Linear , mid:full",
                               "bad", "0:full", "1:full", "-1:huge", "mid:full,mid:linear",
                               "-1:full,-1:linear", "x:full"]
    for spec in specs:
        try:
            theirs = parse_attn_config(spec)
        except ValueError:
            theirs = ValueError
        try:
            ours = attn.parse(spec)
        except ValidationError:
            ours = ValueError
        assert ours == theirs, (spec, ours, theirs)
    print("parser parity: ok")


def check_canonical():
    assert attn.canonical("mid:full,-1:linear") == "-1:linear,mid:full"
    assert attn.canonical("-1:linear,-3:window,mid:full") == "-3:window,-1:linear,mid:full"
    assert attn.canonical({}) == "none"
    assert attn.canonical("none") == "none"
    assert attn.canonical(None) is None
    assert attn.canonical({-1: "linear", "mid": "full"}) == attn.DEFAULT_SPEC
    assert attn.validate(None, [1, 2, 2, 2]) == attn.DEFAULT_SPEC
    assert attn.validate("-4:full,mid:full", [1, 2, 2, 2]) == "-4:full,mid:full"
    for bad in ("-5:full", "-3:linear,mid:full"):
        try:
            attn.validate(bad, [1, 2])
        except ValidationError:
            pass
        else:
            raise AssertionError(f"validate accepted {bad!r} for a 2-level network")
    assert attn.layout_name("mid:full,-1:linear") == "mid-linear"
    assert attn.layout_name("-2:full") is None
    print("canonical + validate: ok")


def check_builds():
    torch.manual_seed(0)
    x, t = torch.randn(1, 3, 64, 64), torch.tensor([10.0])
    specs = [l["spec"] for l in attn.NAMED_LAYOUTS] + README_EXAMPLES
    for depth in (3, 4, 5):
        mults = [1, 2, 2, 2, 4][:depth]
        for spec in specs:
            cfg = attn.parse(spec)
            deepest = min((k for k in cfg if k != "mid"), default=0)
            if -deepest > depth:
                continue
            net = build_unet(CONF_MTYPE, mults, attn_config=cfg)
            with torch.no_grad():
                y = net(x, t)
            assert y.shape == x.shape, (spec, mults, y.shape)
            n_attn = sum(1 for m in net.modules()
                         if type(m).__name__ in ("FullAttention2d", "LinearAttention2d",
                                                 "WindowAttention2d"))
            assert n_attn == len(cfg), (spec, n_attn)
    # the default when nothing is passed is the bottleneck-only layout
    default = build_unet(CONF_MTYPE, [1, 2, 2, 2])
    assert type(default.mid_attn).__name__ == "FullAttention2d"
    assert all(type(m).__name__ == "Identity" for m in default.down_attns)
    # and the old class is untouched by any of this
    old = build_unet("tinyunet_with_attention3", [1, 2, 2, 2])
    assert type(old.mid_attn).__name__ == "SelfAttention2d"
    print("builds: ok")


def _ckpt(net, mults, spec, fmt="upstream", mtype=CONF_MTYPE):
    """A checkpoint recording ``spec`` in one of the formats found in the wild.

    upstream   -- xurdif2.py since a566214, and Kiln's patched trainer: the
                  --attn string as ``attn_conf`` plus the options namespace
    kiln-0914  -- what Kiln wrote for one day before it matched upstream
    None       -- nothing recorded (upstream before a566214)
    """
    state = {"denoise_fn." + k: v for k, v in net.state_dict().items()}
    data = {"step": 0, "model": state, "ema": state, "mults": mults,
            "mtype": mtype, "pred": "x0"}
    if fmt == "upstream":
        data["attn_conf"] = spec
        data["opt"] = argparse.Namespace(model=mtype, mults=mults, pred="x0", attn=spec,
                                         attn_config=attn.parse(spec))
    elif fmt == "kiln-0914":
        data["attn"] = spec
        data["attn_config"] = attn.parse(spec)
    return data


def _refuses(path, phrase):
    try:
        loader.load_net(str(path))
    except ValidationError as e:
        assert phrase in str(e), e
    else:
        raise AssertionError(f"{path.name} loaded although its layout is {phrase!r}")


def check_roundtrip(tmp: Path):
    mults = [1, 2, 2, 2]
    net = build_unet(CONF_MTYPE, mults, attn_config=attn.parse("mid:full,-1:linear"))
    default_net = build_unet(CONF_MTYPE, mults)

    # upstream / Kiln format: spec as typed -> described canonical, loads clean
    p = tmp / "upstream.pt"
    torch.save(_ckpt(net, mults, "mid:full,-1:linear"), p)
    meta = loader.describe(p)
    assert meta.mtype == CONF_MTYPE and meta.attn == "-1:linear,mid:full", meta.attn
    assert meta.to_dict()["attn"] == "-1:linear,mid:full"
    loaded, _ = loader.load_net(str(p))
    assert type(loaded.down_attns[-1]).__name__ == "LinearAttention2d"

    # the options namespace alone is enough
    p2 = tmp / "opt-only.pt"
    d = _ckpt(net, mults, None, fmt=None)
    d["opt"] = argparse.Namespace(model=CONF_MTYPE, attn="-1:linear,mid:full",
                                  attn_config={"mid": "full", -1: "linear"})
    torch.save(d, p2)
    assert loader.describe(p2).attn == "-1:linear,mid:full"
    loader.load_net(str(p2))

    # what Kiln wrote on 14 Sept still reads
    p3 = tmp / "kiln-0914.pt"
    torch.save(_ckpt(net, mults, "-1:linear,mid:full", fmt="kiln-0914"), p3)
    assert loader.describe(p3).attn == "-1:linear,mid:full"
    loader.load_net(str(p3))

    # trained without --attn: recorded as None, which is the bottleneck-only default
    p4 = tmp / "no-attn-flag.pt"
    torch.save(_ckpt(default_net, mults, None), p4)
    assert loader.describe(p4).attn == "mid:full", loader.describe(p4).attn
    loader.load_net(str(p4))

    # nothing recorded, non-default weights (upstream before a566214) -> refused
    p5 = tmp / "unrecorded.pt"
    torch.save(_ckpt(net, mults, None, fmt=None), p5)
    assert loader.describe(p5).attn is None
    _refuses(p5, "records no attention layout")

    # nothing recorded, but the weights really are the default -> fine
    p6 = tmp / "unrecorded-default.pt"
    torch.save(_ckpt(default_net, mults, None, fmt=None), p6)
    loader.load_net(str(p6))

    # a recorded layout that contradicts the weights -> refused
    p7 = tmp / "wrong.pt"
    torch.save(_ckpt(net, mults, "mid:full"), p7)
    _refuses(p7, "records attention layout 'mid:full'")

    # an old attn3 checkpoint carries no layout and takes the old path
    old = build_unet("tinyunet_with_attention3", mults)
    p8 = tmp / "old.pt"
    torch.save(_ckpt(old, mults, None, fmt=None, mtype="tinyunet_with_attention3"), p8)
    m8 = loader.describe(p8)
    assert m8.attn is None and m8.mtype == "tinyunet_with_attention3"
    loader.load_net(str(p8))

    # The options namespace is why resuming must not use torch's default load:
    # from torch 2.6 that is weights-only, and it refuses a Namespace.
    try:
        torch.load(p, weights_only=True)
    except Exception:  # noqa: BLE001 -- UnpicklingError, spelled per torch version
        pass
    else:
        raise AssertionError("a weights-only load accepted the options namespace")
    trainer_src = (ROOT / "vendor" / "xurdif" / "xurdiftrainer.py").read_text(encoding="utf-8")
    assert "torch.load(opt.load, weights_only=False)" in trainer_src, \
        "the vendored trainer resumes with torch's default load, which refuses 'opt'"
    print("checkpoint round trip: ok (upstream, opt-only, 14-Sept Kiln, no --attn, unrecorded)")


if __name__ == "__main__":
    check_parser_parity()
    check_canonical()
    check_builds()
    with tempfile.TemporaryDirectory() as td:
        check_roundtrip(Path(td))
    print("smoke_conf_attention: all ok")
