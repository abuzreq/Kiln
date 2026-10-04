"""Render the example pictures Kiln ships: the Sweep tab's example grids and a
picture of each starter bend preset, all made with one sample model.

    python scripts/make_example_media.py [--model NAME] [--seed N]

The model is one of the sample models (Prepare > Models > Sample models > Get
all), read from the workspace. Every picture goes through the same code the app
uses -- run_sweep for the grids, sampler.run with a bend runtime for the
presets -- at the app's default settings, so an example is what pressing the
button would give on that model and seed.

Writes WebP files to app/frontend/public/examples/ (Vite copies them into the
build) and app/frontend/src/exampleMedia.json, which says what made them. Rerun
it after changing app/frontend/src/sweepExamples.json or a starter in
app/core/craft/starters.py, then rebuild the frontend.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRONTEND = ROOT / "app" / "frontend"
OUT = FRONTEND / "public" / "examples"
MANIFEST = FRONTEND / "src" / "exampleMedia.json"
EXAMPLES = FRONTEND / "src" / "sweepExamples.json"

# The app's own defaults (DEFAULT_SAMPLE_PARAMS in sampleSettings.jsx).
BASE = {"image_size": 512, "steps": 50, "eta": 0.5, "sampler": "dpmpp"}
SHEET_MAX = 1024   # widest a grid is kept; past it the cells stop being legible anyway
PRESET_SIZE = 320  # a preset card shows its picture at under half of this


def axis_values(axis: dict) -> list:
    """The values an axis takes, as axisValues in Sweep.jsx lays them out."""
    if "picks" in axis:
        return list(axis["picks"])
    n = max(2, min(12, int(axis.get("count", 2))))
    if "mult" in axis:
        return [round(axis["from"] * axis["mult"] ** i) for i in range(n)]
    a, b = float(axis["from"]), float(axis["to"])
    # Math.round, not Python's round-half-to-even
    return [int(math.floor(a + (b - a) * i / (n - 1) + 0.5)) for i in range(n)]


def save_webp(img, path: Path, max_w: int):
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), resample=3)
    img.convert("RGB").save(path, "WEBP", quality=82, method=6)
    return f"/examples/{path.name}"


def render(model_path: str, seed: int, bends=None):
    from app.core.craft.bending import build_runtime
    from app.core.engine.sampler import SampleParams, sampler
    from app.core.model_manager import manager

    runtime = None
    if bends:
        meta, backend = manager.describe(model_path)
        runtime = build_runtime(bends, meta, backend=backend)
    params = SampleParams(model_path=model_path, seed=seed, **BASE)
    last = None
    for frame in sampler.run(params, None, None, runtime):
        last = frame
    return last["image_pp"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="txkrs0b-114", help="a sample model's name")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    from app.core.config import workspace
    from app.core.craft import starters
    from app.core.tools.sweep import run_sweep

    model_path = workspace.models / f"{args.model}.pt"
    if not model_path.is_file():
        raise SystemExit(f"{model_path} not found. Get the sample models first: "
                         "Prepare > Models > Sample models > Get all.")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"model": args.model, "seed": args.seed, **BASE, "sweeps": {}, "bends": {}}

    for ex in json.loads(EXAMPLES.read_text(encoding="utf-8")):
        axes = [{"param": a["param"], "values": axis_values(a)}
                for a in (ex["x"], ex.get("y")) if a]
        print(f"sweep {ex['id']}: " + " × ".join(f"{a['param']} {a['values']}" for a in axes),
              flush=True)
        sheet = run_sweep(None, str(model_path), axes, {**BASE, "seed": args.seed})
        manifest["sweeps"][ex["id"]] = save_webp(sheet, OUT / f"sweep-{ex['id']}.webp", SHEET_MAX)

    print("bend: without bends", flush=True)
    plain = render(str(model_path), args.seed)
    manifest["bends"]["plain"] = save_webp(plain, OUT / "bend-plain.webp", PRESET_SIZE)
    for entry in starters.entries():
        print(f"bend: {entry['name']}", flush=True)
        img = render(str(model_path), args.seed, entry["bends"])
        manifest["bends"][entry["name"]] = save_webp(
            img, OUT / f"bend-{entry['name']}.webp", PRESET_SIZE)

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest['sweeps']) + len(manifest['bends'])} pictures to {OUT}")


if __name__ == "__main__":
    main()
