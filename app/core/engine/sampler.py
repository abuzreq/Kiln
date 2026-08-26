"""Guided sampling for xurdif compact diffusion models.

Ported and generalized from the vendored ``xurdifapp3.py`` gradio sampler:
- DDIM sampling with a cosine-beta schedule (via diffusers ``DDIMScheduler``)
- optional CLIP text / image-prompt guidance in x0-space
- init-image mixing (img2img), skip / noise level / "simplify"
- a non-destructive post-processing chain
- per-step streaming (yields a PIL image each denoise step)
- optional model *bending*: a ``BendRuntime`` may inject activation hooks and be
  told the current step so step-scheduled bends fire correctly.

Everything is device-aware (``cuda`` if available, else ``cpu``) so the module can
be imported and exercised without a GPU.
"""
from dataclasses import dataclass, field

from PIL import Image

from app.core.model_manager import manager
from utils.exceptions import EngineError
from utils.logger import get_logger

log = get_logger("sampler")


def _torch():
    import torch

    return torch


def align_size(value: int, mults) -> int:
    """Snap a pixel dimension down to something the UNet can actually process.

    Each level of ``mults`` halves the feature map, and the skip connections
    concatenate, so every dimension must be divisible by ``2 ** len(mults)``
    (16 for the default four-level model). Verified empirically: 768x512 and
    256x384 run fine, 520x510 and 300x300 fail on the concat.
    """
    step = 2 ** len(list(mults or [1, 2, 2, 2]))
    return max(step, int(value) // step * step)


def resolve_seed(raw) -> int:
    """Turn a blank/absent seed into a concrete one.

    Sampling used to fall back to unseeded noise, which made the resulting image
    impossible to regenerate. Every entry point now draws a seed up front and
    reports it back, so any output can be walked back to its run.
    """
    import random

    if raw is None or raw == "":
        return random.randrange(2 ** 31)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return random.randrange(2 ** 31)


def _noise_timestep(t, device):
    """A 1-d timestep tensor for ``scheduler.add_noise``.

    DDIM indexes ``alphas_cumprod`` directly and tolerates a 0-d tensor, but the
    multistep solvers iterate their timesteps (``for t in timesteps``), which
    raises "iteration over a 0-d tensor". Always hand them a 1-element 1-d
    tensor; it broadcasts over the batch either way.
    """
    torch = _torch()
    return torch.tensor([int(t)], device=device)


def pick_device(requested: str | None = None) -> str:
    torch = _torch()
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# Solvers the UI offers. "multistep" ones carry solver history between steps,
# which means the scheduler cannot be rebuilt mid-run (see _apply_updates).
SAMPLERS = {
    "ddim": {
        "label": "DDIM (classic)",
        "cls": "DDIMScheduler",
        "eta": True,
        "multistep": False,
        "resample": True,
        "help": "The original. Predictable, supports Eta and live step changes; needs the most steps.",
    },
    "unipc": {
        "label": "UniPC (fast)",
        "cls": "UniPCMultistepScheduler",
        "eta": False,
        "multistep": True,
        # No state reset is wired for UniPC, so a backward jump would feed it
        # stale history.
        "resample": False,
        "help": "Predictor-corrector solver. Usually matches DDIM quality in about a third of the steps.",
    },
    "dpmpp": {
        "label": "DPM-Solver++ (fast)",
        "cls": "DPMSolverMultistepScheduler",
        "eta": False,
        "multistep": True,
        "resample": True,
        "help": "High-order solver. Very good at 15-25 steps; a close cousin of UniPC.",
    },
    "deis": {
        "label": "DEIS",
        "cls": "DEISMultistepScheduler",
        "eta": False,
        "multistep": True,
        "resample": False,
        "help": "Exponential integrator. Similar speed to DPM-Solver++, slightly different character.",
    },
}
# Not offered: the k-diffusion style solvers (Heun, Euler, DPM2, LMS) are
# sigma-space schedulers. They expect `scale_model_input` and sigma
# conditioning, so dropping them into this alpha/epsilon loop yields a flat
# grey image — verified. Supporting them means a sigma-space rewrite of the
# denoise loop, not a catalogue entry.
# The engine default stays DDIM: cards captured before the sampler existed carry
# no `sampler` field, and they were all rendered with DDIM. Changing this would
# silently alter how every saved recipe replays. New sessions get UniPC via the
# frontend default instead.
DEFAULT_SAMPLER = "ddim"
RECOMMENDED_SAMPLER = "unipc"


def sampler_catalog() -> list[dict]:
    return [
        {"name": k, "label": v["label"], "help": v["help"],
         "eta": v["eta"], "multistep": v["multistep"],
         "resample": v.get("resample", False)}
        for k, v in SAMPLERS.items()
    ]


def sampler_spec(name: str) -> dict:
    return SAMPLERS.get((name or "").lower()) or SAMPLERS[DEFAULT_SAMPLER]


@dataclass
class SampleParams:
    model_path: str
    ema: bool = True
    image_size: int = 512
    steps: int = 50            # DDIM inference steps
    train_steps: int = 1000    # diffusion steps used at train time (schedule length)
    eta: float = 0.5
    skip: int = 0
    seed: int | None = None
    batch_size: int = 1        # multi-seed variations in one GPU run (1–4)
    # guidance
    text: str = ""
    text_weight: float = 0.0
    guidance_step: float = 0.02
    guidance_power: float = 1.0
    spherical: bool = False
    image_prompt_weight: float = 0.0
    cuts: float = 0.5
    # init image
    noise_level: float = 1.0    # 'mul' — how much noise vs init image
    simplify: float = 0.0       # 'weak'
    # output
    postproc: dict = field(default_factory=dict)
    device: str = "auto"
    sampler: str = DEFAULT_SAMPLER
    # RePaint resampling, used by region fill only (see engine/inpaint.py).
    # 1 keeps the single-pass behaviour every existing recipe was made with.
    resample: int = 1
    jump_length: int = 0        # 0 = derive from step count


def _make_betas(timesteps: int):
    """xurdif's cosine betas. Kept as a re-export; the schedule is backend-owned."""
    from app.core.backends.xurdif import cosine_betas

    return cosine_betas(timesteps)


class _Clip:
    """Lazily-loaded CLIP model + cutout sampler for guidance."""

    _inst = None

    @classmethod
    def get(cls, device):
        if cls._inst is None:
            import clip

            model, _ = clip.load("ViT-B/32", device=device, jit=False)
            model = model.eval()
            cls._inst = (model, clip)
        return cls._inst


def _postproc_fn():
    try:
        from ._vendor import ensure_on_path

        ensure_on_path()
        from postproc26 import pprocess

        return pprocess
    except Exception as e:  # noqa: BLE001
        log.info("post-processing unavailable: %s", e)
        return None


RESAMPLE_SOLVERS = tuple(k for k, v in SAMPLERS.items() if v.get("resample"))


def resample_supported(name: str) -> bool:
    """Can this solver survive RePaint's jumps back up the schedule?

    Multistep solvers carry outputs from previous timesteps; a jump invalidates
    them. DPM++ is included because its history can be cleared (see
    ``_reset_solver_state``) at the cost of a first-order step on resume.
    """
    return bool(sampler_spec(name).get("resample"))


def _reset_solver_state(sched):
    """Drop any multistep history so a jump cannot reuse stale outputs."""
    if not hasattr(sched, "model_outputs"):
        return
    order = getattr(sched.config, "solver_order", len(sched.model_outputs))
    sched.model_outputs = [None] * order
    if hasattr(sched, "lower_order_nums"):
        sched.lower_order_nums = 0
    if hasattr(sched, "_step_index"):
        # step() only re-derives the index when it is None; without this the
        # solver would keep counting forward through a schedule that just moved
        # backwards.
        sched._step_index = None


def repaint_positions(n: int, jump_length: int, jump_n_sample: int) -> list[int]:
    """Positions into a descending timestep list, with RePaint's jumps.

    Mirrors ``RePaintScheduler.set_timesteps`` (verified identical across six
    configurations) but yields *positions* rather than absolute timesteps, so it
    applies to whatever grid the chosen solver produced instead of only to
    RePaint's own.

    A position moving *backwards* in the returned list is a jump back up the
    schedule. Do not collapse those into a single solver step -- every
    intermediate timestep has to be re-traversed, or the result is a blank fill.
    """
    jumps = {}
    for j in range(0, n - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1
    out: list[int] = []
    t = n
    while t >= 1:
        t -= 1
        out.append(n - 1 - t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1
                out.append(n - 1 - t)
    return [p for p in out if 0 <= p < n]


def undo_step(sched, x, t, stride, torch):
    """One jump back up the schedule: RePaint Algorithm 1, line 10.

    ``x <- sqrt(1-beta)*x + sqrt(beta)*noise``, applied once per training
    timestep the jump spans.
    """
    last = len(sched.betas) - 1
    for k in range(max(int(stride), 1)):
        beta = sched.betas[min(int(t) + k, last)].to(x.device)
        x = (1 - beta).sqrt() * x + beta.sqrt() * torch.randn_like(x)
    return x


def split_prediction(out, x, alphas_cumprod, t_batch, pred, device):
    """Return ``(eps, x0)`` from a raw model output, whatever the model predicts.

    Both are always needed: the scheduler step is fed epsilon (which is why the
    scheduler's own ``prediction_type`` is always "epsilon" -- see
    ``backends/hfdiffusers``), while x0 is what gets displayed. Shared with the
    region-fill loop in ``engine/inpaint.py`` so the two cannot drift apart.
    """
    torch = _torch()
    out = out.float()
    x_f = x.float()
    ac = alphas_cumprod[t_batch.cpu()].to(device=device, dtype=torch.float32)
    alpha = ac.view(-1, 1, 1, 1)
    beta = (1 - ac).view(-1, 1, 1, 1).clamp(min=1e-8)
    if pred == "eps":
        eps = out
        x0 = (x_f - beta.sqrt() * eps) / alpha.sqrt()
    else:
        x0 = out
        eps = (x_f - alpha.sqrt() * x0) / beta.sqrt()
    return eps.to(x.dtype), x0


class Sampler:
    def _scheduler(self, train_steps: int, steps: int, device: str, name: str = "ddim",
                   schedule=None):
        """Build one of the catalogue's solvers over a backend's noise schedule.

        ``schedule`` is where the two backends genuinely differ. xurdif trained
        every checkpoint on a cosine schedule; Hugging Face DDPM models are
        usually linear. Getting this wrong does not raise -- it just tells the
        model it is at the wrong noise level and quietly degrades the image.
        """
        import diffusers

        spec = sampler_spec(name)
        cls = getattr(diffusers, spec["cls"])
        if schedule is None:
            # Direct callers predating the backend split meant xurdif.
            from app.core.backends.xurdif import XurdifBackend

            schedule = XurdifBackend().schedule(None, train_steps)
        sched = cls(**schedule.scheduler_kwargs())
        sched.set_timesteps(steps, device=device)
        if hasattr(sched, "alphas_cumprod"):
            sched.alphas_cumprod = sched.alphas_cumprod.to(device)
        return sched

    def run(
        self,
        params: SampleParams,
        init_image: Image.Image | None = None,
        image_prompt: Image.Image | None = None,
        bend_runtime=None,
        cancel=None,
        mask: Image.Image | None = None,
        control=None,
        height: int | None = None,
        width: int | None = None,
    ):
        """Generator yielding dicts: {step, total, image, image_pp, images?, images_pp?}.

        ``mask`` (optional, L or RGB) enables masked img2img: denoise inside the
        white region and keep the init image outside, with the mask's gray values
        as a per-pixel blend (feather).

        ``height`` / ``width`` (optional) override the square ``image_size`` so a
        caller can sample at a specific aspect — used by region fill to work at
        the canvas's own resolution. Both must be multiples of ``2 ** len(mults)``
        (see ``align_size``).

        ``control`` (optional) is a Job-like object with ``paused()``,
        ``pop_updates()``, and ``cancelled()`` for cooperative pause/resume and
        live param updates (eta, seed, steps, noise_level).

        When ``params.batch_size`` > 1, each yield also includes ``images`` /
        ``images_pp`` lists (length = batch). ``image`` / ``image_pp`` remain the
        first item for live preview compatibility.
        """
        import time

        torch = _torch()
        device = pick_device(params.device)
        bundle = manager.load(params.model_path, device=device, ema=params.ema)
        model = bundle["model"]
        meta = bundle["meta"]
        backend = bundle["backend"]
        schedule = backend.schedule(bundle["ref"], params.train_steps)
        pred = meta.pred

        eta = float(params.eta)
        noise_level = float(params.noise_level)
        steps = int(params.steps)
        skip = min(int(params.skip), steps)
        bs = max(1, min(4, int(params.batch_size or 1)))

        spec = sampler_spec(params.sampler)
        sched = self._scheduler(params.train_steps, steps, device, params.sampler, schedule)
        timeline = list(sched.timesteps[skip:])
        stride = max(1, int(sched.config.num_train_timesteps) // max(steps, 1))
        # A "plan" is the order positions in `timeline` are visited. Normally
        # that is 0,1,2,... — RePaint resampling revisits earlier positions so
        # the model can reconcile a fill with its surroundings, which is the one
        # thing that makes the walk non-monotonic.
        resample = max(1, int(getattr(params, "resample", 1) or 1))
        repaint = resample > 1 and mask is not None
        if repaint:
            if not resample_supported(params.sampler):
                raise EngineError(
                    f"'{params.sampler}' cannot resample: it carries solver state "
                    f"between steps, which a jump back up the schedule would "
                    f"invalidate. Use {' or '.join(RESAMPLE_SOLVERS)}.")
            jump_length = int(getattr(params, "jump_length", 0) or 0)
            if jump_length <= 0:
                # RePaintScheduler only creates jump points in
                # range(0, n - jump_length, jump_length), so a fixed 10 gives one
                # jump point at 20 steps and none at 10. The paper uses 250
                # steps; Kiln runs 15-50, so scale with the step count.
                jump_length = max(2, len(timeline) // 5)
            jump_length = max(1, min(jump_length, max(len(timeline) - 1, 1)))
            plan = repaint_positions(len(timeline), jump_length, resample)
        else:
            plan = list(range(len(timeline)))
        remaining = list(timeline)
        done = 0
        total = sum(1 for a, b in zip([-1] + plan, plan) if b > a)

        # ``height``/``width`` let a caller work at the canvas's own aspect (region
        # fill); plain generation still uses the square ``image_size``.
        H = int(height or params.image_size)
        W = int(width or params.image_size)
        init_noise = _batched_init_noise(torch, bs, H, W, device, params.seed)

        orig = None
        mask_t = None
        if init_image is not None:
            img = init_image.convert("RGB").resize((W, H))
            import torchvision.transforms.functional as TF

            orig = TF.to_tensor(img).to(device).unsqueeze(0) * 2 - 1  # 1,3,H,W
            if bs > 1:
                orig = orig.expand(bs, -1, -1, -1).contiguous()
            x = noise_level * sched.add_noise(
                orig, init_noise, _noise_timestep(timeline[0], device)
            )
        else:
            x = noise_level * init_noise * sched.init_noise_sigma

        if mask is not None and orig is not None:
            m = mask.convert("L").resize((W, H), Image.BILINEAR)
            import torchvision.transforms.functional as TF

            mask_t = TF.to_tensor(m).to(device).unsqueeze(0)  # 1,1,H,W in [0,1]
            if bs > 1:
                mask_t = mask_t.expand(bs, -1, -1, -1).contiguous()
            # start fully noised only inside the mask
            noised = noise_level * sched.add_noise(
                orig, init_noise, _noise_timestep(timeline[0], device)
            )
            x = mask_t * noised + (1 - mask_t) * orig

        # optional guidance setup
        use_guidance = (params.text and params.text_weight > 0) or (
            image_prompt is not None and params.image_prompt_weight > 0
        )
        clip_ctx = None
        txt_enc = None
        if use_guidance:
            try:
                clip_model, clip_mod = _Clip.get(device)
                clip_ctx = clip_model
                if params.text and params.text_weight > 0:
                    tok = clip_mod.tokenize(params.text).to(device)
                    txt_enc = clip_model.encode_text(tok).detach().float()
            except Exception as e:  # noqa: BLE001
                log.warning("guidance disabled: %s", e)
                use_guidance = False

        if bend_runtime is not None:
            bend_runtime.attach(model)
            bend_runtime.set_total(total)

        pprocess = _postproc_fn()
        tensor_to_pil = __import__("torchvision").transforms.ToPILImage()

        def _is_cancelled():
            if cancel is not None and cancel():
                return True
            if control is not None and control.cancelled():
                return True
            return False

        def _apply_updates(updates: dict):
            nonlocal eta, noise_level, steps, remaining, total, sched, skip
            if not updates:
                return
            if "eta" in updates and updates["eta"] is not None:
                eta = float(updates["eta"])
            if "noise_level" in updates and updates["noise_level"] is not None:
                noise_level = float(updates["noise_level"])
            if "seed" in updates:
                seed = updates["seed"]
                if seed is not None and seed != "":
                    torch.manual_seed(int(seed))
            if "steps" in updates and updates["steps"] is not None and not spec["multistep"]:
                # Rebuilding a multistep scheduler would silently drop its solver
                # history and corrupt the run, so step count is fixed for those.
                new_steps = max(int(updates["steps"]), done)
                if new_steps != steps:
                    steps = new_steps
                    skip = min(int(params.skip), steps)
                    sched = self._scheduler(params.train_steps, steps, device, params.sampler,
                                            schedule)
                    full = list(sched.timesteps[skip:])
                    remaining = full[done:]
                    total = done + len(remaining)
                    if bend_runtime is not None:
                        bend_runtime.set_total(total)

        try:
            pos_prev = -1
            plan_i = 0
            while plan_i < len(plan):
                if _is_cancelled():
                    break

                if control is not None:
                    while control.paused() and not _is_cancelled():
                        time.sleep(0.1)
                    if _is_cancelled():
                        break
                    _apply_updates(control.pop_updates())
                    if plan_i >= len(plan):
                        break

                pos = plan[plan_i]
                plan_i += 1
                if pos <= pos_prev:
                    # A jump back up the schedule. No model evaluation, so it
                    # does not advance progress; the solver's multistep history
                    # (if any) is now stale and has to go.
                    x = undo_step(sched, x, timeline[pos], stride, torch)
                    _reset_solver_state(sched)
                    pos_prev = pos
                    continue
                pos_prev = pos

                i = timeline[pos]
                remaining = timeline[pos + 1:]
                # Bends are scheduled over the descending schedule position, not
                # the evaluation counter: with jumps the latter is not monotonic
                # in t, and a scheduled bend would fire erratically.
                step_idx = pos if repaint else done
                if bend_runtime is not None:
                    bend_runtime.set_step(step_idx)

                # Model gets a per-batch timestep vector; DDIMScheduler.step must get a
                # scalar — a batched timestep hits `if prev_timestep >= 0` and raises
                # "Boolean value of Tensor with more than one value is ambiguous".
                t_batch = torch.full((bs,), int(i), device=device, dtype=torch.long)
                t_step = int(i)
                with torch.no_grad():
                    autocast = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if device == "cuda"
                        else _nullcontext()
                    )
                    with autocast:
                        out = model(x, t_batch)
                    eps, x0 = split_prediction(
                        out, x, sched.alphas_cumprod, t_batch, pred, device)
                    # Only DDIM-style solvers take eta, and the multistep ones do
                    # not return pred_original_sample — but we already computed x0
                    # above, so use that and stay scheduler-agnostic.
                    kw = {"eta": eta} if spec["eta"] else {}
                    s = sched.step(eps, t_step, x, **kw)
                    x = s["prev_sample"].detach()
                    x_s = x0.detach()

                    if mask_t is not None and orig is not None:
                        x_s = mask_t * x_s + (1 - mask_t) * orig
                        if remaining:
                            t_next = remaining[0]
                            noised = noise_level * sched.add_noise(
                                orig, torch.randn_like(orig),
                                _noise_timestep(t_next, device),
                            )
                            x = mask_t * x + (1 - mask_t) * noised
                        else:
                            x = mask_t * x + (1 - mask_t) * orig
                    elif remaining and abs(noise_level - 1.0) > 1e-6:
                        # Extra/less noise on non-final unmasked steps.
                        # At noise_level == 1.0 this branch is skipped (unchanged).
                        t_next = remaining[0]
                        ac_next = sched.alphas_cumprod[
                            torch.tensor([int(t_next)], device="cpu")
                        ].to(device=device, dtype=torch.float32)
                        beta_next = (1 - ac_next).view(-1, 1, 1, 1).clamp(min=1e-8)
                        x = x + (noise_level - 1.0) * beta_next.sqrt() * torch.randn_like(x)

                done += 1
                raws, pps = self._to_images(x_s, params, pprocess, tensor_to_pil, backend, meta)
                out = {
                    "step": done,
                    "total": max(total, done),
                    "image": raws[0],
                    "image_pp": pps[0],
                }
                if bs > 1:
                    out["images"] = raws
                    out["images_pp"] = pps
                yield out
        finally:
            if bend_runtime is not None:
                bend_runtime.detach()

    def _to_images(self, x_s, params, pprocess, tensor_to_pil, backend=None, meta=None):
        """Convert a (B,3,H,W) tensor to parallel lists of raw / postproc PILs.

        Mapping x0 into displayable range is the backend's call: xurdif's models
        are low-contrast and have always been stretched to a fixed std, which
        would be a distortion applied to anything else.
        """
        if backend is None:
            from app.core.backends.xurdif import XurdifBackend

            backend = XurdifBackend()
        im = backend.to_display(x_s, meta)

        raws = [tensor_to_pil(im[b].cpu()) for b in range(im.shape[0])]
        if pprocess is None or not params.postproc:
            return raws, list(raws)

        pps = []
        for b in range(im.shape[0]):
            try:
                o = _PostprocOpts(params.postproc)
                single = im[b : b + 1] * 2 - 1
                pim = pprocess(single, o)
                pim = pim - pim.min()
                pim = pim / pim.max().clamp(min=1e-6)
                pps.append(tensor_to_pil(pim[0].cpu()))
            except Exception as e:  # noqa: BLE001
                log.info("postproc failed: %s", e)
                pps.append(raws[b])
        return raws, pps


def _batched_init_noise(torch, bs, H, W, device, seed):
    """Independent Gaussian noise per batch item; honors base seed when set."""
    if seed is None:
        return torch.zeros(bs, 3, H, W, device=device).normal_(0, 1)
    out = torch.empty(bs, 3, H, W, device=device)
    for i in range(bs):
        torch.manual_seed(int(seed) + i)
        out[i] = torch.randn(3, H, W, device=device)
    return out


class _PostprocOpts:
    """Adapts a plain dict to the attribute-style opts the vendored postproc expects."""

    _DEFAULTS = dict(
        postproc=True, onorm=True, contrast=1.0, saturation=1.0, gamma=1.0,
        eqhist=0.0, unsharp=0.0, c1=0.0, c2=1.0, sharpenlast=True, sharpkernel=3,
        median=0, ovl0=0, noise=0.0, bil=0, bils1=75, bils2=75,
    )

    def __init__(self, overrides: dict):
        for k, v in self._DEFAULTS.items():
            setattr(self, k, v)
        for k, v in (overrides or {}).items():
            setattr(self, k, v)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


sampler = Sampler()


def postprocess_only(image: Image.Image, opts: dict) -> Image.Image:
    """Re-apply post-processing to an already generated image (no re-sampling)."""
    pprocess = _postproc_fn()
    if pprocess is None:
        return image
    torch = _torch()
    import torchvision.transforms.functional as TF

    t = TF.to_tensor(image.convert("RGB")).unsqueeze(0) * 2 - 1
    try:
        o = _PostprocOpts(opts)
        t = pprocess(t, o)
        t = t - t.min()
        t = t / t.max().clamp(min=1e-6)
        return TF.to_pil_image(t[0])
    except Exception as e:  # noqa: BLE001
        raise EngineError(f"post-processing failed: {e}")
