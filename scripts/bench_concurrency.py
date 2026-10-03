"""Does running generations side by side on one GPU buy anything?

The generation queue design (docs/generation-queue-design.md) lets runs on
*different* models overlap. Whether that is worth more than a latency nicety
depends on one number this measures: wall time for N runs in N threads against
the same N runs back to back, in one process and one CUDA context -- exactly
how Kiln's sampling threads share the GPU.

Each case is timed three ways. Frames render on first read (``sampler.Frame``),
so what a run costs depends on how often its caller looks, and that CPU work
runs under the GIL:

    every  -- the caller reads every step's image
    10hz   -- reads at most every 0.1 s, as the perform routes' live preview does
    none   -- reads only the final image, as previews, sweeps and the explorer do

Threaded cases run twice: as Kiln's threads do today, all on the device's
default CUDA stream, and with a stream per thread (see ``_case``).

Only plain runs are timed (no guidance, no bends); one CLIP-guided pair is added
for its peak VRAM. Two runs on one model are timed back to back, on two copies of
the model at once, and as a batch of 2 (``--same-only`` runs just those).

    python scripts/bench_concurrency.py
    python scripts/bench_concurrency.py --models a.pt b.pt --sizes 128 256 512 --same-only
    python scripts/bench_concurrency.py --models a.pt b.pt c.pt --sizes 256 --steps 20
    python scripts/bench_concurrency.py --json bench.json
"""
import argparse
import json
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app.core.engine import sampler as sampler_mod  # noqa: E402
from app.core.engine.sampler import SampleParams, align_size, sampler  # noqa: E402
from app.core.model_manager import manager  # noqa: E402

# Different architectures on purpose: two attn3 nets of the same shape would
# flatter the overlap, since their kernels are identical.
DEFAULT_MODELS = [
    Path.home() / "kiln" / "models" / "txplsa-200.pt",         # attn3, 1,2,2,2
    Path.home() / "kiln" / "models" / "galaxies-step4000.pt",  # conf attn, 1,2,2,4
    Path.home() / "kiln" / "models" / "tnxsyn-28.pt",          # attn3, 1,2,4
]

RENDER_MODES = ("every", "10hz", "none")


# --- second copies of a model -----------------------------------------------------
# Kiln never runs two jobs on one model at once. The two-copy case asks whether
# a second copy of the net, on its own stream, would make that worth doing; it
# measured 1.03-1.14x (docs/generation-queue-design.md), so the app has no such
# thing and the benchmark brings its own. A thread inside using_copy(i > 0) gets
# its own copy, loaded from the file, when the sampler asks the manager.
_copy = threading.local()
_copies: dict = {}
_shared_load = manager.load


def _load(path, device="cpu", ema=True):
    i = getattr(_copy, "i", 0)
    if not i:
        return _shared_load(path, device=device, ema=ema)
    key = (str(path), device, ema, i)
    if key not in _copies:
        from app.core import backends

        backend, ref = backends.resolve(str(path))
        model, meta = backend.load(ref, device=device, ema=ema)
        _copies[key] = {"model": model, "meta": meta, "backend": backend, "ref": ref}
    return _copies[key]


manager.load = _load


@contextmanager
def using_copy(i: int):
    prev = getattr(_copy, "i", 0)
    _copy.i = i
    try:
        yield
    finally:
        _copy.i = prev
PREVIEW_INTERVAL = 0.1  # matches app/backend/routes/perform.py


# --- rendering modes ----------------------------------------------------------
# How often a run's caller reads a frame's image; set per case, read by _run.
_mode = "every"


def set_render_mode(mode: str):
    global _mode
    _mode = mode


# --- one run --------------------------------------------------------------------
def _params(model, size, steps, sampler_name, seed, batch=1, text=""):
    meta = manager.load(str(model), device="cuda")["meta"]
    return SampleParams(
        model_path=str(model), image_size=align_size(size, meta.mults), steps=steps,
        eta=0.5, seed=seed, batch_size=batch, sampler=sampler_name, device="cuda",
        postproc={}, text=text,
    )


def _run(params) -> float:
    t0 = time.perf_counter()
    shown = 0.0
    last = None
    for frame in sampler.run(params):
        last = frame
        now = time.perf_counter()
        if _mode == "every" or (_mode == "10hz" and now - shown >= PREVIEW_INTERVAL):
            shown = now
            frame["image_pp"]
    if last is not None:
        last["image_pp"]                     # every caller wants the result
    # This thread's stream only: a device-wide sync would make each thread wait
    # for the others and blur the per-run times.
    torch.cuda.current_stream().synchronize()
    return time.perf_counter() - t0


_streams: list = []


def _stream(i: int):
    """The i-th side stream, created once. The caching allocator keeps a pool per
    stream, so a fresh stream per case grows reserved VRAM until Windows starts
    paging it -- which first showed up here as a 5x slowdown, not an error."""
    while len(_streams) <= i:
        _streams.append(torch.cuda.Stream())
    return _streams[i]


def _case(param_sets, parallel: bool, streams: bool = False, copies: bool = False) -> dict:
    """Wall time and peak VRAM for a set of runs, back to back or one thread each.

    Threads in one process share the device's default CUDA stream unless told
    otherwise, so their kernels queue behind each other whatever the Python
    side does. ``streams`` gives each thread its own, which is the only way the
    GPU can actually run two models' kernels at once. ``copies`` gives run i
    copy i of its model, as a generation lane does for two runs on one model.
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    per_run = [None] * len(param_sets)
    errors = []

    def work(i, p):
        try:
            with using_copy(i if copies else 0):
                if streams:
                    with torch.cuda.stream(_stream(i)):
                        per_run[i] = _run(p)
                else:
                    per_run[i] = _run(p)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    t0 = time.perf_counter()
    if parallel:
        threads = [threading.Thread(target=work, args=(i, p)) for i, p in enumerate(param_sets)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for i, p in enumerate(param_sets):
            work(i, p)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {
        "wall": wall,
        "per_run": per_run,
        "peak_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
        "error": errors[0] if errors else None,
    }


def _median_case(param_sets, parallel, repeats, streams=False, copies=False) -> dict:
    runs = [_case(param_sets, parallel, streams, copies) for _ in range(repeats)]
    bad = next((r for r in runs if r["error"]), None)
    if bad:
        return bad
    return {
        "wall": statistics.median(r["wall"] for r in runs),
        "per_run": [statistics.median(r["per_run"][i] for r in runs)
                    for i in range(len(param_sets))],
        "peak_mib": max(r["peak_mib"] for r in runs),
        "reserved_mib": max(r["reserved_mib"] for r in runs),
        "error": None,
    }


# --- the benchmark ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", help="two or three model paths (default: a mixed set)")
    ap.add_argument("--sizes", nargs="+", type=int, default=[256, 512])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--sampler", default="ddim")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--modes", nargs="+", default=list(RENDER_MODES), choices=RENDER_MODES)
    ap.add_argument("--no-guided", action="store_true", help="skip the CLIP-guided pair")
    ap.add_argument("--same-only", action="store_true",
                    help="only the same-model cases: sequential, two copies, batch of 2")
    ap.add_argument("--json", help="also write the raw results here")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs CUDA: concurrency on the CPU says nothing about the GPU")
        return 1

    models = [Path(m) for m in (args.models or DEFAULT_MODELS)]
    models = [m for m in models if m.exists()]
    if len(models) < 2:
        print("need at least two existing models; pass --models")
        return 1
    models = models[:3]

    # Every model stays resident for the whole run: the app's cache holds two,
    # and a reload mid-case would be timed as if it were sampling.
    manager.MAX_LOADED = 16
    for m in models:
        manager.load(str(m), device="cuda")
    with using_copy(1):                    # A's second copy, for the two-copy case
        manager.load(str(models[0]), device="cuda")

    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{free / 2 ** 30:.2f} of {total / 2 ** 30:.2f} GiB free at start")
    print(f"torch {torch.__version__}, {args.sampler}, {args.steps} steps, "
          f"median of {args.repeats}")
    for i, m in enumerate(models):
        meta = manager.load(str(m), device="cuda")["meta"]
        print(f"  {'ABC'[i]}: {m.name}  ({meta.backend} {meta.mtype} {meta.mults})")
    print()

    # Warm up every model at every size (cudnn autotune, allocator growth), in
    # every render mode, so the first timed case is not paying for it.
    for size in args.sizes:
        for m in models:
            for mode in args.modes:
                set_render_mode(mode)
                _run(_params(m, size, 3, args.sampler, 1))
            with using_copy(1):
                _run(_params(models[0], size, 3, args.sampler, 1))

    results = []
    header = (f"| size | render | case | wall s | speedup | per-run s | peak MiB | reserved MiB |\n"
              f"|---|---|---|---|---|---|---|---|")
    print(header)

    def report(size, mode, name, res, base=None):
        row = {"size": size, "render": mode, "case": name, **res}
        if base is not None and not res["error"] and not base["error"]:
            row["speedup"] = base["wall"] / res["wall"]
        results.append(row)
        if res["error"]:
            print(f"| {size} | {mode} | {name} | - | - | {res['error'][:60]} | - | - |")
            return
        sp = f"{row['speedup']:.2f}x" if "speedup" in row else ""
        per = " / ".join(f"{t:.2f}" for t in res["per_run"])
        print(f"| {size} | {mode} | {name} | {res['wall']:.2f} | {sp} | {per} "
              f"| {res['peak_mib']:.0f} | {res['reserved_mib']:.0f} |", flush=True)

    def same_model_cases(size, mode):
        """Two runs on one model: back to back, on two copies at once, and batched.

        Two copies on their own streams is what a lane can do for queued runs
        whose settings differ; a batch needs identical settings but no copy.
        """
        a = _params(models[0], size, args.steps, args.sampler, 7)
        b = _params(models[0], size, args.steps, args.sampler, 8)
        seq_same = _median_case([a, b], False, args.repeats)
        report(size, mode, "A,A sequential", seq_same)
        report(size, mode, "A|A' two copies+streams",
               _median_case([a, b], True, args.repeats, streams=True, copies=True), seq_same)
        batch = _params(models[0], size, args.steps, args.sampler, 7, batch=2)
        report(size, mode, "A batch of 2", _median_case([batch], False, args.repeats),
               seq_same)

    for size in args.sizes:
        for mode in args.modes:
            set_render_mode(mode)
            if args.same_only:
                same_model_cases(size, mode)
                continue
            two = [_params(m, size, args.steps, args.sampler, 7) for m in models[:2]]
            seq2 = _median_case(two, False, args.repeats)
            report(size, mode, "A,B sequential", seq2)
            report(size, mode, "A|B threads", _median_case(two, True, args.repeats), seq2)
            report(size, mode, "A|B threads+streams",
                   _median_case(two, True, args.repeats, streams=True), seq2)

            if len(models) >= 3:
                three = [_params(m, size, args.steps, args.sampler, 7) for m in models]
                seq3 = _median_case(three, False, args.repeats)
                report(size, mode, "A,B,C sequential", seq3)
                report(size, mode, "A|B|C threads", _median_case(three, True, args.repeats), seq3)
                report(size, mode, "A|B|C threads+streams",
                       _median_case(three, True, args.repeats, streams=True), seq3)

            same_model_cases(size, mode)

    if not args.no_guided:
        set_render_mode("every")
        size = min(args.sizes)
        sampler_mod._Clip.get("cuda")      # load once, outside the threads
        sampler_mod._Clip.cutter()
        guided = [_params(m, size, args.steps, args.sampler, 7, text="a detailed painting")
                  for m in models[:2]]
        _run(guided[0])                    # warm up the guidance path
        seq_g = _median_case(guided, False, args.repeats)
        report(size, "every", "guided A,B sequential", seq_g)
        report(size, "every", "guided A|B threads",
               _median_case(guided, True, args.repeats), seq_g)
        report(size, "every", "guided A|B threads+streams",
               _median_case(guided, True, args.repeats, streams=True), seq_g)

    set_render_mode("every")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "gpu": torch.cuda.get_device_name(0),
            "models": [str(m) for m in models],
            "steps": args.steps, "sampler": args.sampler, "repeats": args.repeats,
            "results": results,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
