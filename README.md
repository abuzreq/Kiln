# Kiln

A desktop application for small-data training, sampling, and model manipulation with compact diffusion models. Inspired by [Autolume](https://metacreation-lab.github.io/autolume/), Kiln organizes workflows into **Workshop** (data preparation, training, model management) and **Play** (sampling, layer manipulation, merging, parameter sweeps).

Kiln supports two backends:

1. **xurdif:** The compact diffusion engine by [Hannu Töyrylä](https://github.com/htoyryla/xurdif).
2. **Diffusers:** Hugging Face's `UNet2DModel` / `TinyUNet2DModel` ecosystem for unconditional pixel-space models (DDPM), including LoRA support.

Typical workflow: **Data → Train → Models → Play**.

---

## Features

* **Workshop ▸ Data:** Image and video framing (crop, pad, stretch, non-square), data augmentations, and live dataset preview.
* **Workshop ▸ Train:** Train from scratch or fine-tune existing models. Includes loss curves, live sample generation, and snapshot saves.
* **Workshop ▸ Models:** Model library, checkpoint pinning, Hugging Face model imports, and format conversion.
* **Play ▸ Create:** DDIM sampling with step-by-step previews, CLIP text/image guidance, img2img, inpainting (soft/hard brushes), and upscaling.
* **Play ▸ Bend:** Interactive UNet layer activation targeting, hook-based ops, scheduled hooks, and seed-matched A/B comparison.
* **Play ▸ Merge:** Two-way model weight merging with live comparison against parent models.
* **Play ▸ Sweep:** 1D and 2D parameter sweep grids with exportable contact sheets.

---

## Requirements

* **Python 3.10+**
* **Git** (required on system `PATH` to fetch dependencies like OpenAI CLIP)
* **NVIDIA GPU with CUDA** for sampling and training (xurdif requires CUDA; Diffusers can run on CPU, but training will be slow).
* **Node.js 18+** (only needed to build frontend changes; prebuilt UI assets are included in `app/frontend/build`).
* **Disk Space:** ~6 GB (including CUDA PyTorch wheels and dependencies).

---

## Installation & Quick Start

Launchers handle virtual environment setup automatically:

* **Windows:** Run `kiln.bat`
* **macOS:** Run `kiln.command`
* **Linux:** Run `./kiln.sh`

#### Linux notes

The launcher builds its own virtual environment, but three things have to come
from the distribution first. Kiln tells you which one is missing when it hits
them; installing them up front saves the round trip.

```bash
# Debian / Ubuntu
sudo apt install python3-venv python3-pip git   # venv is a separate package here
sudo apt install libgl1 libglib2.0-0            # OpenCV links these
sudo apt install python3-gi gir1.2-webkit2-4.1  # optional: the desktop window
```

Without the WebKit packages everything still works — Kiln prints its URL and you
open it in a browser, which is also what `./kiln.sh --no-window` does. If the
launcher lost its executable bit (downloading a zip rather than cloning does
this), `chmod +x kiln.sh` restores it.

### Manual Setup

If you prefer setting up environments manually:

```bash
# 1. Run the setup script with system Python
python install.py

# 2. Launch the backend using the virtual environment
# Windows
.venv\Scripts\python.exe start.py

# macOS / Linux
.venv/bin/python start.py

```

### PyTorch CUDA Manual Install

The installer attempts to download the CUDA-enabled wheel automatically. To install or override manually:

```bash
# Windows
.venv\Scripts\python.exe -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Linux / macOS
.venv/bin/python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

```

*(For RTX 50-series cards, replace `cu121` with `cu124` or `cu128`.)*

### CLI Options & Configuration

* `--no-window`: Run in headless mode (prints local URL to terminal).
* `--port <int>`: Set backend port (defaults to `8777` or `$KILN_PORT`).
* `KILN_WORKSPACE`: Set root storage path for datasets, models, and outputs (defaults to `~/kiln`).

---

## Acquiring Models

Repositories do not include checkpoint weights. Models can be sourced via:

1. **Pretrained xurdif Checkpoints:** Download `.pt` files from the [author's Dropbox repository](https://www.dropbox.com/scl/fo/flh4pczukrrlb3ar1rfuc/AAT22M2b21Tf1yKe3Ji0HS0?rlkey=f1zdhexy36p3hffcun686m77c&dl=0) and load them in **Workshop ▸ Models ▸ Get a model**.
2. **Hugging Face Hub:** Import unconditional DDPM models under **Workshop ▸ Models ▸ From Hugging Face**. *(Note: Latent text-to-image models like Stable Diffusion are not supported.)*
3. **Local Training:** Prepare an image folder under **Workshop ▸ Data** and run training via **Workshop ▸ Train**.

---

## Development

Run the API backend and the Vite development server concurrently:

```bash
# Terminal 1: Backend API
.venv/bin/python start.py --no-window

# Terminal 2: Frontend (Vite dev server at localhost:5199, proxies API to :8777)
cd app/frontend
npm install
npm run dev

```

### Smoke Tests

Test scripts live in `scripts/`:

```bash
python scripts/smoke_engine.py
python scripts/smoke_craft.py
python scripts/smoke_merge.py
python scripts/smoke_diffusers.py
python scripts/smoke_train.py             # xurdif tests require CUDA
python scripts/smoke_tinyunet_parity.py   # Verifies TinyUNet Diffusers/xurdif parity
python scripts/smoke_golden.py --check    # Checkpoint hashing regression test

```

---

## Project Structure

```
app/
├── backend/       # Flask REST API and file helpers
├── core/          # Training pipelines, sampling, hooks, and model logic
│   └── backends/  # Engine implementations (xurdif / hfdiffusers)
└── frontend/      # React client (Vite)
vendor/xurdif/     # Vendored xurdif engine
models/            # Local scratch checkpoints

```

---

## Backend Engine Comparison

| Capability | xurdif | Diffusers |
| --- | --- | --- |
| **Model Format** | `.pt` checkpoints | `UNet2DModel`, `TinyUNet2DModel` |
| **Sampling / Painting / Bending** | Yes | Yes |
| **Weight Merging** | Yes (matching layer dimensions) | Yes (matching network configs) |
| **Training From Scratch** | Yes (CUDA required) | Yes |
| **Fine-tuning** | Resume run | Full fine-tune |
| **LoRA Support** | No | Yes (via PEFT) |
| **Loss Function** | Edge-weighted L1 (optional) + SSIM | MSE or edge-weighted L1 |
| **Noise Schedule** | Fixed cosine | Set via `scheduler_config.json` |

### `TinyUNet2DModel` Port

`TinyUNet2DModel` is a direct reimplementation of xurdif's architecture inside the Diffusers ecosystem. At 512×512 resolution, it evaluates ~4× faster during sampling and ~7× faster during training than a standard `UNet2DModel` while cutting VRAM usage in half.

To convert a legacy `.pt` model to Diffusers format without precision loss, use **Workshop ▸ Models ▸ Re-home a model**.

---

## License

MIT. See [LICENSE](https://www.google.com/search?q=LICENSE).

Kiln vendors the `xurdif` library by Hannu Töyrylä (MIT License). Upstream tracking and modifications are documented in [vendor/xurdif/UPSTREAM.md](https://www.google.com/search?q=vendor/xurdif/UPSTREAM.md).