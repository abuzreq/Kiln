# Kiln

**A creativity-support tool for small-data training and crafting of image-generating diffusion models.**

Kiln wraps the compact [xurdif](https://github.com/htoyryla/xurdif) diffusion engine
in a desktop app organised around **Studio** (datasets, train, models) and
**Play** (sample, paint, bend, merge, sweep) — inspired by
[Autolume](https://metacreation-lab.github.io/autolume/). Model *crafting* is
first-class: interactively **bend** a model's activations (targeting specific
layers on a live map) and **merge** two compatible models, then save and reapply
those bends and recipes across your work.

Typical flow: **Data → Train → Models → Play**.

---

## Highlights

- **Studio ▸ Datasets** — import images/videos, frame (crop/pad/stretch, non-square),
  augment, and build reusable datasets with a live preview.
- **Studio ▸ Train** — start a run from scratch or from a library model; revisit past
  runs; watch live samples and the loss curve; save snapshots into the library.
- **Studio ▸ Models** — named library models, training snapshots, pinning, rename,
  and a “Get a model” downloader.
- **Play ▸ Sample** — guided DDIM sampling with per-step preview, CLIP text/image
  guidance, init-image (img2img), post-process, and upscale.
- **Play ▸ Paint** — mask a region and restyle it; soft and hard brushes.
- **Play ▸ Bend** — a hook-based op catalog, a live UNet map, scheduled bends,
  and same-seed compare of samples with vs without bends.
- **Play ▸ Merge** — strictly **2-way** merges with same-seed A / B / recipe compare.
- **Play ▸ Sweep** — vary one or two sampling parameters and compare 1D/2D results
  in-panel; download a contact sheet PNG of the whole grid.
- **Library** — drawer for named models, saved bends, merge recipes, and captures.

## Requirements

- **Python 3.10+**
- **An NVIDIA GPU with CUDA** for training and sampling (the xurdif engine is
  CUDA-only). The app itself, dataset prep, model introspection, bending and
  merging also work on CPU.
- Node 18+ is only needed if you want to rebuild the frontend; a prebuilt bundle
  is committed under `app/frontend/build`.

## Quick start

From the project folder:

- **Windows:** double-click `kiln.bat`
- **macOS:** double-click `kiln.command`
- **Linux:** run `./kiln.sh`

The launcher creates a virtual environment on first run and opens the app.
Later launches skip `pip` when requirements have not changed (a fingerprint of
the requirements files plus a cheap import probe — no PyTorch import). To force
a reinstall: `python install.py --reinstall`.

On first install, if an NVIDIA GPU is detected the installer pulls a **CUDA**
build of PyTorch (~2.5 GB) rather than the CPU-only wheel PyPI serves by
default — the engine cannot train or sample without it. If that ever needs
doing by hand:

```bash
.venv/bin/pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If the app's device badge reads **CPU only**, hover it: Kiln reports whether the
cause is a CPU-only PyTorch build, a driver problem, or no GPU at all.

To run manually:

```bash
python install.py        # one-time setup
python start.py          # launch (add --no-window to use a browser)
```

The workspace (source, datasets, runs, models, captures, library) defaults to
`~/kiln` and can be changed in-app. Legacy `~/kiln/projects/<name>/` folders are
still read so existing data is not lost; new work writes to the workspace root.

## Development

```bash
# backend
python start.py --no-window          # serves API + built frontend at :8777

# frontend (hot reload, proxies /api to the backend)
cd app/frontend && npm install && npm run dev
```

Dev smoke tests (CPU-only) live in `scripts/`:

```bash
python scripts/smoke_engine.py
python scripts/smoke_craft.py
python scripts/smoke_api_craft.py
python scripts/smoke_merge.py
python scripts/smoke_serve.py
python scripts/smoke_full.py
```

## Layout

```
app/
  backend/     Flask API (routes + data helpers)
  core/        training, sampling, craft, library, tools
  frontend/    React UI (Vite); build/ is what start.py serves
vendor/xurdif/ vendored engine (see vendor/xurdif/UPSTREAM.md)
models/        optional local pretrained / fine-tuned seeds
```

## License

See [LICENSE](LICENSE).
