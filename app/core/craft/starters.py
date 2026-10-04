"""Bend recipes that ship with Kiln.

Bending is the one part of Kiln with no safe default: the ops all do something,
but a first-time stack is usually either invisible or a grey mess, and the map of
layer names says nothing about what pulling on one *looks* like. These are six
starting points chosen so that each isolates a single, legible effect -- load
one, press Compare, and the difference is obvious enough to reason from.

Every recipe targets *groups* ("encoder", "attention", ...) rather than concrete
layer names, so they carry across architectures and backends. A stack saved with
resolved layer names would only work on the model it was made on.

These are read-only: they are merged into the Library listing by the /craft/bends
route and cannot be edited or deleted in place. Saving a stack under the same
name shadows the built-in with the user's own copy, which is the intended way to
adapt one.
"""

# step_start / step_end are fractions of the run (0 = first denoise step, 1 =
# last), matching BendRuntime in bending.py.
STARTERS: list[dict] = [
    {
        "name": "starter-soft-focus",
        "notes": "Damps the encoder, so the model commits less firmly to structure. "
                 "Softer, hazier, less insistent images. The gentlest thing here -- "
                 "start with it, then try 0.4 or 0.9 to feel the range.",
        "bends": [{
            "op": "multiply", "params": {"value": 0.65},
            "targets": ["encoder"], "step_start": 0.0, "step_end": 1.0, "active": True,
        }],
    },
    {
        "name": "starter-etched",
        "notes": "Mixes a little of the decoder's own edges back into it. Shapes break into "
                 "outlined, cell-like patches, like leaded glass or an etching. A little goes "
                 "a long way: edge strength has no sign, so past about 0.2 the picture "
                 "washes out to grey (measured on three sample models; 0.6 left only grey).",
        "bends": [{
            "op": "gradient", "params": {"mix": 0.08},
            "targets": ["decoder"], "step_start": 0.0, "step_end": 1.0, "active": True,
        }],
    },
    {
        "name": "starter-bottleneck-flare",
        "notes": "Amplifies the middle of the network, where the model decides what the "
                 "picture is about rather than what it looks like. Composition shifts "
                 "more than texture does. Targets the mid block as well as attention on "
                 "purpose: these compact models carry only a few small attention modules, "
                 "and scaling them alone barely registers (measured: 1.8 vs 20 for both).",
        "bends": [{
            "op": "multiply", "params": {"value": 2.0},
            "targets": ["attention", "mid"], "step_start": 0.0, "step_end": 1.0, "active": True,
        }],
    },
    {
        "name": "starter-grain-storm",
        "notes": "Injects noise into the decoder for most of the run. Flat areas break up "
                 "into grain while the composition holds. Ends before the last steps so "
                 "the model still resolves some of what the noise suggested. In the "
                 "bottleneck the same noise barely showed: the decoder rebuilt over it.",
        "bends": [{
            "op": "noise", "params": {"std": 0.3, "seed": 0},
            "targets": ["decoder"], "step_start": 0.2, "step_end": 0.9, "active": True,
        }],
    },
    {
        "name": "starter-poster",
        "notes": "Clips the decoder's range, so tone collapses into flat bands instead "
                 "of gradients. Screen-print feel. Narrow the min/max together for "
                 "harder posterisation.",
        "bends": [{
            "op": "clamp", "params": {"min": -0.8, "max": 0.8},
            "targets": ["decoder"], "step_start": 0.0, "step_end": 1.0, "active": True,
        }],
    },
    {
        "name": "starter-drift",
        "notes": "Slides encoder features sideways over the first half of the run, so "
                 "structure and detail end up disagreeing about where things are. "
                 "Smeared, double-exposed results. Two bends: see how a stack composes.",
        "bends": [
            {"op": "roll", "params": {"dx": 6, "dy": 0},
             "targets": ["encoder"], "step_start": 0.0, "step_end": 0.5, "active": True},
            {"op": "multiply", "params": {"value": 1.2},
             "targets": ["decoder"], "step_start": 0.5, "step_end": 1.0, "active": True},
        ],
    },
]

STARTER_NAMES = frozenset(s["name"] for s in STARTERS)


def entries() -> list[dict]:
    """The starters as Library-shaped entries, flagged so the UI can group them."""
    out = []
    for s in STARTERS:
        bends = [dict(b, id=f"{s['name']}-{i}") for i, b in enumerate(s["bends"])]
        out.append({
            "name": s["name"],
            "bends": bends,
            "notes": s["notes"],
            "model_hint": "",
            "builtin": True,
        })
    return out


def is_starter(name: str) -> bool:
    return name in STARTER_NAMES
