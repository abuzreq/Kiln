"""Smoke test for the configurable-attention architecture (CPU, no GPU needed).

Holds Kiln's pure-Python spec parser to the vendored one, builds every named
layout across network depths, round-trips the layout through a checkpoint, and
checks that a conf checkpoint with an unrecorded or wrong layout is refused
rather than silently loaded with missing keys.

    python scripts/smoke_conf_attention.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _ckpt(net, mults, spec, **extra):
    state = {"denoise_fn." + k: v for k, v in net.state_dict().items()}
    data = {"step": 0, "model": state, "ema": state, "mults": mults,
            "mtype": CONF_MTYPE, "pred": "x0"}
    if spec is not None:
        data["attn"] = spec
        data["attn_config"] = attn.parse(spec)
    data.update(extra)
    return data


def check_roundtrip(tmp: Path):
    mults = [1, 2, 2, 2]
    net = build_unet(CONF_MTYPE, mults, attn_config=attn.parse("mid:full,-1:linear"))

    # Kiln-written: both keys, spec as typed -> described canonical, loads clean
    p = tmp / "kiln.pt"
    torch.save(_ckpt(net, mults, "mid:full,-1:linear"), p)
    meta = loader.describe(p)
    assert meta.mtype == CONF_MTYPE and meta.attn == "-1:linear,mid:full", meta.attn
    assert meta.to_dict()["attn"] == "-1:linear,mid:full"
    loaded, _ = loader.load_net(str(p))
    assert type(loaded.down_attns[-1]).__name__ == "LinearAttention2d"

    # attn_config only (the dict xurdiffer26c.py reads) is enough
    p2 = tmp / "dict-only.pt"
    d = _ckpt(net, mults, None)
    d["attn_config"] = {"mid": "full", -1: "linear"}
    torch.save(d, p2)
    assert loader.describe(p2).attn == "-1:linear,mid:full"
    loader.load_net(str(p2))

    # neither key: upstream-trained with a non-default layout -> refused loudly
    p3 = tmp / "unrecorded.pt"
    torch.save(_ckpt(net, mults, None), p3)
    assert loader.describe(p3).attn is None
    try:
        loader.load_net(str(p3))
    except ValidationError as e:
        assert "records no attention layout" in str(e), e
    else:
        raise AssertionError("an unrecorded non-default layout loaded silently")

    # neither key, but the weights really are the constructor default -> fine
    p4 = tmp / "unrecorded-default.pt"
    torch.save(_ckpt(build_unet(CONF_MTYPE, mults), mults, None), p4)
    loader.load_net(str(p4))

    # a recorded layout that lies about the weights -> refused
    p5 = tmp / "wrong.pt"
    torch.save(_ckpt(net, mults, "mid:full"), p5)
    try:
        loader.load_net(str(p5))
    except ValidationError as e:
        assert "records attention layout 'mid:full'" in str(e), e
    else:
        raise AssertionError("a wrong recorded layout loaded silently")

    # an old attn3 checkpoint carries no layout and takes the old path
    old = build_unet("tinyunet_with_attention3", mults)
    p6 = tmp / "old.pt"
    torch.save({**_ckpt(old, mults, None), "mtype": "tinyunet_with_attention3"}, p6)
    m6 = loader.describe(p6)
    assert m6.attn is None and m6.mtype == "tinyunet_with_attention3"
    loader.load_net(str(p6))
    print("checkpoint round trip: ok")


if __name__ == "__main__":
    check_parser_parity()
    check_canonical()
    check_builds()
    with tempfile.TemporaryDirectory() as td:
        check_roundtrip(Path(td))
    print("smoke_conf_attention: all ok")
