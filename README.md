# Kiln

Kiln is a desktop app for training and playing with small diffusion models on your own images.
Train a compact model on a few hundred pictures, then sample, paint, bend, merge and sweep what
it learned. Everything runs locally: your datasets, models and images stay on your machine.

Inspired by [Autolume](https://metacreation-lab.github.io/autolume/), Kiln is organised into two
spaces:

* **Prepare** — datasets, training runs, and your model library.
* **Create** — a layered canvas, model bending, merging, and parameter sweeps.

The usual path is **Data → Train → Models → Create**, though a downloaded model lets you start in
Create on day one.

---

## The app

### Start

The hub you land on. Two buttons lead into **Prepare** and **Create**, and a bar shows any job
running in either. Below them, every model in your library appears as a card that cross-fades
through sample images Kiln generated for it in the background; clicking one opens it in Create at
that image's seed. Saved captures sit along the bottom, and clicking one restores the settings and
model that made it.

### Prepare ▸ Data

A dataset is Kiln's record of images you already have, not a copy of them.

* **Add a folder or file** by path and it is read where it lives, subfolders included. New files
  dropped into that folder join the dataset on their own. **Upload…** is the one exception: files
  sent from the browser are copied in, since they have nowhere else to live.
* **Videos** are turned into frames on request, at a rate you choose.
* **Remove** an image and it leaves the dataset but stays on disk; "Show removed" and **Restore
  all** undo it. Removing a source folder never touches the folder.
* **Framing and augmentation** decide what training sees: a size, how images are fitted to it
  (center crop, stretch, or pad), and which augmentations to use — H flip, V flip, Rotate,
  Brightness, Contrast. Each augmentation offers its own options, and they combine, so the recipe
  says exactly how big the training set is: *H flip ×2 · Rotate ×4 = 8 versions of each image.
  Training will see 2,560 images per pass (320 × 8).* Nothing is random, and no augmented copies
  are written to disk.
* The **Preview** panel shows that whole set for the first image, updating as you edit.
* **Train on this dataset** carries it straight to a new run.

### Prepare ▸ Train

**Train new model** sets up a run: a dataset, a name, and whether to start from scratch or from a
model in your library. Presets suit the run to your GPU, and Kiln estimates peak VRAM before you
start. Everything else — image size, batch, learning rate, architecture, attention layout,
snapshot frequency — sits under Advanced.

While a run goes, you get live samples, a loss curve, the exact command in use, and the log.
Training can be stopped and later continued from its latest checkpoint. Snapshots are training
checkpoints, not library models, until you **Save** one into the library under a name.

**View previous runs** lists what you have already trained, with its samples, loss and snapshots.

### Prepare ▸ Models

Your library, plus three ways to add to it:

* **Get a model** — download a checkpoint from a URL.
* **Re-home a model** — convert a xurdif `.pt` into the Diffusers format, leaving the original
  alone.
* **From Hugging Face** — check a repo for compatibility and import it. Unconditional
  pixel-space models only; latent text-to-image models such as Stable Diffusion are not supported.

Model cards show a sample, the architecture, attention layout and training step, and offer **Use
in Create**, **Train**, rename and pin. Models inside Kiln's workspace can be deleted; models Kiln
merely found elsewhere are hidden instead, leaving the file where it is. A checkbox lists hidden
models again.

### Create ▸ Canvas

Sampling on a layered canvas. Pick a model, set steps and seed, and generate — the image builds
step by step in front of you, and can be paused or stopped and kept as it is.

* **Change** controls how far a generation moves away from what is already on the canvas, which is
  how img2img and repainting work.
* **Masks** are painted, drawn as shapes, grown from a wand, split from the image itself, or
  generated as patterns; a masked generation fills only that region, with feathering and a
  harmonize pass to settle the seam.
* **CLIP guidance** steers sampling towards a prompt or a reference image.
* **Finish** applies contrast, gamma and sharpening, and upscales.
* Layers, saved assets, and a Results history sit alongside; any result can become a layer or be
  saved.
* Downloaded PNGs carry their settings inside them, so opening one later can restore the run.

Shortcuts: `Ctrl+Z` / `Ctrl+Shift+Z` undo and redo, `Ctrl+D` clears the mask, `Ctrl+Shift+I`
inverts it, and space-drag pans the canvas.

### Create ▸ Bend

Bending rewrites the network's activations while an image forms. It changes the output, not the
model file. Pick layers on a map of the U-Net, stack operations with their own amount and a
window of the run to apply over, and compare the bent result against the plain one on the same
seed. Setups can be saved as presets, exported and imported, and a single bend parameter can be
swept into an animation.

### Create ▸ Merge

Blend two compatible models, with per-block weights for encoder, middle and decoder. Kiln samples
both parents and the merge on the same seed so you can see what the blend did, and the result can
be saved as a new library model.

### Create ▸ Sweep

Lay a parameter out across a grid — one axis or two — sample every cell, and download the contact
sheet.

### Discoveries

Along the bottom of Create sits a background explorer. Given a model, it tries random bends and
keeps the samples that look new to it, judged by CLIP or DINOv2 similarity. It yields to your own
generations, so it fills the strip while you work. Any discovery can be loaded into Bend, saved as
a bend preset, or opened in a map that lets you walk between similar results.

---

## Engines

Kiln runs two backends, and the screens work the same way for both.

| | xurdif | Diffusers |
| --- | --- | --- |
| **Model format** | `.pt` checkpoints | `UNet2DModel`, `TinyUNet2DModel` |
| **Sampling, painting, bending** | Yes | Yes |
| **Merging** | Yes (matching layer sizes) | Yes (matching configs) |
| **Training from scratch** | Yes (needs CUDA) | Yes |
| **Fine-tuning** | Continue a run | Full fine-tune |
| **LoRA** | No | Yes (via PEFT) |
| **Loss** | Edge-weighted L1 (optional) + SSIM | MSE or edge-weighted L1 |
| **Noise schedule** | Fixed cosine | `scheduler_config.json` |

**xurdif** is the compact diffusion engine by [Hannu Töyrylä](https://github.com/htoyryla/xurdif),
vendored in `vendor/xurdif/`. **Diffusers** is Hugging Face's ecosystem for unconditional
pixel-space models.

`TinyUNet2DModel` is a reimplementation of xurdif's architecture inside Diffusers. At 512×512 it
samples about 4× faster and trains about 7× faster than a standard `UNet2DModel`, at half the
VRAM. **Prepare ▸ Models ▸ Re-home a model** converts a `.pt` into it without precision loss.

---

## Requirements

* **Python 3.10+**
* **Git** on your `PATH` (some dependencies, such as OpenAI CLIP, are fetched from source)
* **An NVIDIA GPU with CUDA.** xurdif requires it; Diffusers will run on CPU, but training will
  be slow.
* **~6 GB of disk** for the CUDA PyTorch wheels and the rest of the dependencies.
* **Node.js 18+** only if you want to change the interface. The built UI ships in
  `app/frontend/build`.

## Install and run

The launchers build the virtual environment on first run and start the app:

* **Windows:** `kiln.bat`
* **macOS:** `kiln.command`
* **Linux:** `./kiln.sh`

To reach Kiln from another machine on the same network, use `start_lan.bat`, `start_lan.command`
or `./start_lan.sh` instead; each prints the address to open. Kiln has no login, so anyone who can
reach that address can browse your datasets, models and files — only do this on a network you
trust.

Kiln keeps everything it makes in one workspace folder, `~/kiln` by default. The **Workspace**
button in the top bar shows what is in it and can move it somewhere else.

### Linux notes

Three things have to come from your distribution first. Kiln says which one is missing when it
hits it, but installing them up front saves the round trip:

```bash
# Debian / Ubuntu
sudo apt install python3-venv python3-pip git   # venv is a separate package here
sudo apt install libgl1 libglib2.0-0            # OpenCV links these
sudo apt install python3-gi gir1.2-webkit2-4.1  # optional: the desktop window
```

Without the WebKit packages everything still works — Kiln prints its URL and you open it in a
browser, which is what `./kiln.sh --no-window` does anyway. If the launcher lost its executable
bit (downloading a zip rather than cloning does that), `chmod +x kiln.sh` restores it.

### Setting up by hand

```bash
python install.py                 # system Python; builds .venv and installs everything

.venv\Scripts\python.exe start.py   # Windows
.venv/bin/python start.py           # macOS / Linux
```

### If it says CPU but you have a GPU

The installer reads your NVIDIA driver version and picks the newest PyTorch build that driver can
run. Installing a build *newer* than the driver is the usual cause of a GPU that PyTorch cannot
see. To repair it:

```bash
python install.py --fix-torch
```

To pick one by hand, find your driver with `nvidia-smi` and use the matching index:

| Driver | Build | Index |
| --- | --- | --- |
| 580+ | CUDA 13.0 | `cu130` |
| 575.51+ | CUDA 12.9 | `cu129` |
| 570+ | CUDA 12.8 | `cu128` (lowest for RTX 50-series) |
| 525.60+ | CUDA 12.6 | `cu126` |
| 450.80+ | CUDA 11.8 | `cu118` |

```bash
# Windows
.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Linux / macOS
.venv/bin/python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Command line options

* `--no-window` — run headless and print the URL.
* `--port <int>` — backend port (default `8777`, or `$KILN_PORT`).
* `--lan` — serve to the local network as well; shorthand for `--host 0.0.0.0`.
* `--host <addr>` — bind a specific address (default `127.0.0.1`, this machine only, or
  `$KILN_HOST`).
* `KILN_WORKSPACE` — workspace folder (default `~/kiln`).

## Getting models

The repository ships no weights. Three ways to get some:

1. **Pretrained xurdif checkpoints** from the [author's Dropbox folder](https://www.dropbox.com/scl/fo/flh4pczukrrlb3ar1rfuc/AAT22M2b21Tf1yKe3Ji0HS0?rlkey=f1zdhexy36p3hffcun686m77c&dl=0),
   loaded under **Prepare ▸ Models ▸ Get a model**.
2. **Hugging Face**, under **Prepare ▸ Models ▸ From Hugging Face**.
3. **Train your own**, which is what the rest of Prepare is for.

---

## Development

```bash
# backend
.venv/bin/python start.py --no-window

# frontend: Vite on :5199, proxying the API to :8777
cd app/frontend
npm install
npm run dev
```

`npm run build` writes the bundle in `app/frontend/build`, which is what the launchers serve.

### Smoke tests

`scripts/` holds runnable checks rather than a unit-test suite. Each one builds a throwaway
workspace and prints what it verified:

```bash
python scripts/smoke_engine.py            # the engine layer, on CPU
python scripts/smoke_craft.py             # layer introspection, bend ops, hooks
python scripts/smoke_merge.py             # two-way merge through the API
python scripts/smoke_diffusers.py         # the Diffusers backend, on CPU
python scripts/smoke_train.py             # training on both engines (xurdif needs CUDA)
python scripts/smoke_datasets.py          # linked datasets, safe delete, augmentation sets
python scripts/smoke_library.py           # preview failures, delete vs hide for models
python scripts/smoke_tinyunet_parity.py   # TinyUNet matches xurdif layer for layer
python scripts/smoke_golden.py --check    # sampling regression, by checkpoint hash
python scripts/smoke_cuda_pick.py         # driver to PyTorch build choice (no GPU needed)
```

### Layout

```
app/
├── backend/       # Flask API: routes and the data layer
├── core/          # sampling, training, bending, the model library
│   └── backends/  # xurdif and hfdiffusers engine implementations
└── frontend/      # React client (Vite)
utils/             # shared helpers
scripts/           # smoke tests
vendor/xurdif/     # vendored xurdif engine
models/            # checkpoints that ship or are downloaded locally
```

---

## License

MIT — see [LICENSE](LICENSE).

Kiln vendors the `xurdif` library by Hannu Töyrylä (MIT). What was changed and why is recorded in
[vendor/xurdif/UPSTREAM.md](vendor/xurdif/UPSTREAM.md).
