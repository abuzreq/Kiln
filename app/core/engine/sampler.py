"""Guided sampling for xurdif compact diffusion models.

Ported and generalized from the vendored ``xurdifapp3.py`` gradio sampler:
- DDIM sampling with a cosine-beta schedule (via diffusers ``DDIMScheduler``)
- optional CLIP text / image-prompt guidance in x0-space (see ``guide_step``)
- init-image mixing (img2img), skip / noise level / "simplify"
- a non-destructive post-processing chain
- per-step streaming (yields a PIL image each denoise step)
- optional model *bending*: a ``BendRuntime`` may inject activation hooks and be
  told the current step so step-scheduled bends fire correctly.

Everything is device-aware (``cuda`` if available, else ``cpu``) so the module can
be imported and exercised without a GPU.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial

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
# DPM-Solver++ over UniPC: it is the only fast solver whose multistep history can
# be reset (see _reset_solver_state), which is what lets a region fill resample
# (RePaint's jumps back up the schedule). UniPC matches it on plain generation
# and then cannot harmonize a fill at all, so recommending it meant most people
# met inpainting through the solver least able to do it.
RECOMMENDED_SAMPLER = "dpmpp"


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
    # Only ever a *balance* against an image prompt. It cannot act as an on/off
    # switch or a strength: the guidance gradient is RMS-normalised before the
    # step (see ``guide_step``), which divides a lone text weight straight back
    # out -- 10 and 90 give byte-identical images. A non-empty ``text`` is what
    # turns text guidance on, and ``guidance_step`` is how hard it pulls.
    text_weight: float = 1.0
    guidance_step: float = 0.02
    guidance_power: float = 1.0
    spherical: bool = False
    image_prompt_weight: float = 0.0
    cuts: float = 0.5
    # init image
    noise_level: float = 1.0    # 'mul' — how much noise vs init image
    simplify: float = 0.0       # 'weak'
    # Scales the *initial* noise only (1.0 = untouched). Applies to every start
    # the sampler can make -- pure noise, an init image's added noise, and the
    # noise a mask is filled with -- because all three are drawn from the same
    # tensor. Below 1 the run starts with less variance than the model was
    # trained to expect, which reads as calmer, flatter, more washed-out output
    # (and, with an init image, as less departure from it).
    attenuation: float = 1.0
    # output
    postproc: dict = field(default_factory=dict)
    device: str = "auto"
    sampler: str = DEFAULT_SAMPLER
    # RePaint resampling, used by region fill only (see engine/inpaint.py).
    # 1 keeps the single-pass behaviour every existing recipe was made with.
    resample: int = 1
    jump_length: int = 0        # 0 = derive from step count


# Guidance settings a paused run can be resumed with. Kept together because the
# whole set is re-read at once (see ``_build_guidance``) -- changing any one of
# them means re-encoding, so there is no point tracking them separately.
GUIDANCE_KEYS = (
    "text", "text_weight", "guidance_step", "guidance_power", "spherical",
    "cuts", "image_prompt_weight",
)


def _make_betas(timesteps: int):
    """xurdif's cosine betas. Kept as a re-export; the schedule is backend-owned."""
    from app.core.backends.xurdif import cosine_betas

    return cosine_betas(timesteps)


class _Clip:
    """Lazily-loaded CLIP model + cutout sampler for guidance."""

    _inst = None
    _cutter = None

    @classmethod
    def get(cls, device):
        torch = _torch()
        if cls._inst is None:
            import clip

            model, _ = clip.load("ViT-B/32", device=device, jit=False)
            cls._inst = (model.eval(), clip)
            return cls._inst
        model, clip_mod = cls._inst
        want = torch.device(device).type
        if next(model.parameters()).device.type != want:
            # The cache outlives a device switch -- a CPU run after a CUDA one,
            # or the reverse. clip.load keeps fp16 weights on GPU and casts to
            # fp32 on CPU (where half is slow and partly unimplemented), so
            # match that rather than only moving the tensors.
            model = model.to(device)
            model = model.float() if want == "cpu" else model.half()
            cls._inst = (model, clip_mod)
        return cls._inst

    @classmethod
    def cutter(cls):
        """The vendored GPU cutout sampler that CLIP is scored over.

        CLIP sees 224px crops, never the frame itself. Scoring one downscaled
        copy of the image would guide composition and nothing else; a spread of
        random crops is what lets a prompt reach local detail. ``cuts`` slides
        between the two -- 0 is many small crops (detail), 1 is few large ones
        (structure).
        """
        if cls._cutter is None:
            from ._vendor import ensure_on_path

            ensure_on_path()
            from cutouts25 import CutoutConfig, GpuCutoutSampler

            cls._cutter = GpuCutoutSampler(CutoutConfig())
        return cls._cutter


def spherical_dist_loss(x, y):
    """Great-circle distance between two CLIP embeddings.

    The alternative to cosine distance. Same minimum, but its gradient does not
    flatten as the embeddings converge, so a strong prompt keeps pulling right
    to the end of the run instead of stalling once it is roughly satisfied.
    """
    import torch.nn.functional as F

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def _clip_cutouts(img01, cuts: float):
    """CLIP-normalised crops of a (B,3,H,W) image in [0,1], gradients intact."""
    torch = _torch()
    cutter = _Clip.cutter()
    # GpuCutoutSampler asserts a single image, so batch items are cut separately
    # and stacked. The losses stay effectively per-item anyway: each crop's
    # gradient flows back only into the item it was cut from.
    return torch.cat(
        [cutter.sample(img01[b:b + 1], slider=float(cuts)) for b in range(img01.shape[0])],
        dim=0,
    )


def _encode_cutouts(clip_model, crops):
    """Encode crops with CLIP, whatever precision the model was loaded at.

    ``clip.load`` keeps fp16 weights on CUDA and casts to fp32 only on CPU, so
    handing it a float32 batch raises a dtype mismatch on GPU. The embedding
    comes back as fp32 either way -- the guidance math wants the headroom.
    """
    dtype = next(clip_model.parameters()).dtype
    return clip_model.encode_image(crops.to(dtype)).float()


def _embed_loss(target_enc, img_enc, spherical: bool):
    torch = _torch()
    if spherical:
        return spherical_dist_loss(target_enc, img_enc).mean()
    return (1 - torch.cosine_similarity(target_enc, img_enc)).mean()


def _text_weight(params) -> float:
    """The prompt's weight against an image prompt, never zero.

    A prompt is switched on by existing, so a stored 0 (the old default, and
    what any recipe written before that carried) must not silently mute it.
    """
    w = float(getattr(params, "text_weight", 1.0) or 0.0)
    return w if w > 0 else 1.0


def clip_grad(x0, clip_model, params, txt_enc=None, imgp_enc=None):
    """``dL/dx0`` for the active CLIP losses, or None if nothing is guiding.

    The gradient stops at x0; it is deliberately *not* backpropagated through
    the UNet into x. Stepping x0 and re-deriving epsilon from it (see
    ``guide_step``) points the same way at a fraction of the memory, and it
    leaves the model's forward pass free to run under fp16 autocast -- with
    bending hooks attached -- without any of that landing in an autograd graph.
    """
    torch = _torch()
    with torch.enable_grad():
        x0 = x0.detach().float().requires_grad_(True)
        # CLIP wants [0,1]; x0 lives in [-1,1] and can overshoot early on.
        crops = _clip_cutouts((x0.clamp(-1, 1) + 1) * 0.5, params.cuts)
        img_enc = _encode_cutouts(clip_model, crops)

        loss = None
        for enc, weight in ((txt_enc, _text_weight(params)),
                            (imgp_enc, params.image_prompt_weight)):
            if enc is None or float(weight) <= 0:
                continue
            term = float(weight) * _embed_loss(enc, img_enc, params.spherical)
            loss = term if loss is None else loss + term
        if loss is None:
            return None
        grad = torch.autograd.grad(loss, x0)[0]
    return torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)


def guide_step(x, eps, x0, alpha, params, clip_model, txt_enc=None, imgp_enc=None):
    """Nudge x0 toward the prompt, then re-derive the epsilon the solver is fed.

    Working in x0-space is what keeps this solver-agnostic: every scheduler in
    the catalogue takes epsilon, so guidance has to end as a modified epsilon
    rather than as a direct edit of x.

    Two knobs shape the step. ``guidance_step`` is how far to move along the
    gradient, which is RMS-normalised first -- raw CLIP gradient magnitudes vary
    by orders of magnitude between prompts, and without normalising, a value
    that works for one prompt tears another apart. ``guidance_power`` ramps that
    step with sqrt(alpha_bar), the signal strength: at 0 guidance is constant
    across the run, and the higher it goes the longer guidance holds off. Early
    steps are nearly pure noise, and pushing CLIP hard against noise bakes in
    its adversarial texture rather than the subject.
    """
    torch = _torch()
    grad = clip_grad(x0, clip_model, params, txt_enc, imgp_enc)
    if grad is None:
        return eps, x0
    rms = grad.flatten(1).pow(2).mean(dim=1).sqrt().view(-1, 1, 1, 1) + 1e-8
    scale = float(params.guidance_step) * alpha.sqrt() ** float(params.guidance_power)
    x0 = x0 - scale * (grad / rms)         # descent: the losses are distances
    beta = (1 - alpha).clamp(min=1e-8)
    eps = ((x.float() - alpha.sqrt() * x0) / beta.sqrt()).to(eps.dtype)
    return eps, x0


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


def alpha_bar(alphas_cumprod, t_batch, device):
    """``ᾱ`` for a batch of timesteps, shaped (B,1,1,1) and always fp32.

    The one place the schedule is read, so the denoise loop and CLIP guidance
    cannot end up disagreeing about the noise level of the step they are on.
    """
    torch = _torch()
    ac = alphas_cumprod[t_batch.cpu()].to(device=device, dtype=torch.float32)
    return ac.view(-1, 1, 1, 1)


def split_prediction(out, x, alphas_cumprod, t_batch, pred, device):
    """Return ``(eps, x0)`` from a raw model output, whatever the model predicts.

    Both are always needed: the scheduler step is fed epsilon (which is why the
    scheduler's own ``prediction_type`` is always "epsilon" -- see
    ``backends/hfdiffusers``), while x0 is what gets displayed. Shared with the
    region-fill loop in ``engine/inpaint.py`` so the two cannot drift apart.
    """
    out = out.float()
    x_f = x.float()
    alpha = alpha_bar(alphas_cumprod, t_batch, device)
    beta = (1 - alpha).clamp(min=1e-8)
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
        """Generator yielding a ``Frame`` per denoise step.

        A frame reads like the dict this used to yield --
        ``{step, total, batch, image, image_pp, images?, images_pp?}`` -- but its
        images are rendered on first read (see ``Frame``), so a caller pays only
        for the steps it actually looks at.

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
        # Guidance is set up *before* the seeded noise is drawn, and the order
        # matters: loading CLIP allocates random tensors, so doing it after
        # torch.manual_seed advanced the global stream by a different amount on
        # the first guided run of a process than on every one after it. Same
        # seed, same settings, different image -- verified, and reproducibility
        # is the one promise every capture in Kiln makes.
        # Guidance is rebuilt from scratch whenever its settings change, rather
        # than patched, so a mid-run prompt swap cannot leave a stale embedding
        # paired with a new weight. Encoding a prompt is milliseconds next to a
        # denoise step, so there is nothing to save by being cleverer.
        def _build_guidance(p):
            """``(clip_model, txt_enc, imgp_enc)`` for ``p``, or a triple of None."""
            wants_text = bool((p.text or "").strip())
            wants_image = image_prompt is not None and p.image_prompt_weight > 0
            if not (wants_text or wants_image):
                return None, None, None
            try:
                clip_model, clip_mod = _Clip.get(device)
                t_enc = None
                if wants_text:
                    tok = clip_mod.tokenize(p.text).to(device)
                    t_enc = clip_model.encode_text(tok).detach().float()
                i_enc = None
                if wants_image:
                    import torchvision.transforms.functional as TF

                    ip = TF.to_tensor(
                        image_prompt.convert("RGB").resize((W, H))
                    ).to(device).unsqueeze(0).clamp(0, 1)
                    with torch.no_grad():
                        # Cut the reference the same way the sample will be cut,
                        # or the two embeddings describe different things.
                        i_enc = _encode_cutouts(
                            clip_model, _clip_cutouts(ip, p.cuts)).detach()
                return clip_model, t_enc, i_enc
            except Exception as e:  # noqa: BLE001
                log.warning("guidance disabled: %s", e)
                return None, None, None

        # Guidance reads its settings from here, not from ``params``: a paused run
        # can be resumed with a different prompt, and the caller's params (which
        # the recipe card was built from) must not change under it.
        gparams = params
        clip_ctx, txt_enc, imgp_enc = _build_guidance(gparams)
        guided = clip_ctx is not None and (txt_enc is not None or imgp_enc is not None)

        init_noise = _batched_init_noise(torch, bs, H, W, device, params.seed)
        # Attenuation scales the seed noise once, here, before it is used. Every
        # way a run can start draws from this tensor -- pure noise below, an init
        # image's added noise, and the noise a mask is filled with -- so scaling
        # it covers all of them without a branch per case.
        atten = 1.0 if params.attenuation is None else float(params.attenuation)
        atten = max(0.0, min(1.0, atten))
        if abs(atten - 1.0) > 1e-6:
            init_noise = init_noise * atten

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
            nonlocal gparams, clip_ctx, txt_enc, imgp_enc, guided
            if not updates:
                return
            # Guidance settings are live: a run can be paused, given a different
            # prompt, and resumed onto it. An empty prompt or a zero weight turns
            # guidance off the same way, so "" is a meaningful value here and only
            # None means "not being set".
            live_guidance = {k: updates[k] for k in GUIDANCE_KEYS
                             if k in updates and updates[k] is not None}
            if live_guidance:
                from dataclasses import replace

                gparams = replace(gparams, **live_guidance)
                clip_ctx, txt_enc, imgp_enc = _build_guidance(gparams)
                guided = clip_ctx is not None and (
                    txt_enc is not None or imgp_enc is not None)
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
                    if guided:
                        # Guidance edits x0 and hands back the epsilon implied by
                        # the edit, so the solver step below is untouched by it.
                        eps, x0 = guide_step(
                            x, eps, x0,
                            alpha_bar(sched.alphas_cumprod, t_batch, device),
                            gparams, clip_ctx, txt_enc, imgp_enc,
                        )
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
                yield Frame(
                    step=done, total=max(total, done), batch=bs, x=x_s,
                    render=partial(self._to_images, params=params, pprocess=pprocess,
                                   tensor_to_pil=tensor_to_pil, backend=backend, meta=meta),
                )
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


class Frame(Mapping):
    """One denoise step's output, with its images rendered on first read.

    The sampler used to convert every step's x0 to PIL -- the backend's display
    mapping, ``ToPILImage`` and the whole post-processing chain, per batch item --
    while the routes publish a preview at most ten times a second and every
    other caller reads only the last frame. Measured with
    scripts/bench_concurrency.py, the discarded renders were 12-35% of a 256px
    run. A frame now holds the step's x0 and renders when someone reads an
    image, once; ``step``, ``total`` and ``batch`` cost nothing. Reading
    ``image`` / ``image_pp`` renders only the first batch item -- all a live
    preview shows -- and ``images`` / ``images_pp`` render the rest. The display
    mapping and post-processing are per item, so item 0 comes out the same
    either way.

    It is a read-only mapping so every ``frame["image"]`` / ``frame.get(...)``
    caller works unchanged. A caller that wants to transform the images -- region
    fill composites them onto the canvas -- uses ``map_images``, which defers
    the work along with the render instead of forcing it.

    Rendering draws no random numbers unless post-processing's ``noise`` option
    is set, which the app never sets, so when a frame renders cannot change the
    sampled image.
    """

    _SINGLE = ("image", "image_pp")
    _BATCH = ("images", "images_pp")

    def __init__(self, *, step: int, total: int, batch: int, x, render):
        self._meta = {"step": step, "total": total, "batch": batch}
        self._x = x                    # the step's x0, (B,3,H,W); dropped once rendered
        self._render = render          # x -> (raws, pps)
        self._maps = []
        self._first = None             # (raw, pp) of item 0
        self._all = None               # (raws, pps) of every item

    def map_images(self, fn) -> "Frame":
        """Apply ``fn`` to every image this frame yields, raw and processed alike."""
        self._maps.append(fn)
        if self._first is not None:
            self._first = tuple(fn(im) for im in self._first)
        if self._all is not None:
            self._all = tuple([fn(im) for im in ims] for ims in self._all)
        return self

    def _image_keys(self):
        return self._SINGLE + (self._BATCH if self._meta["batch"] > 1 else ())

    def _rendered(self, x):
        raws, pps = self._render(x)
        for fn in self._maps:
            raws = [fn(im) for im in raws]
            pps = [fn(im) for im in pps]
        return raws, pps

    def _ensure(self, every: bool):
        if self._all is not None or (not every and self._first is not None):
            return
        if every or self._meta["batch"] == 1:
            self._all = self._rendered(self._x)
            self._first = (self._all[0][0], self._all[1][0])
            # The step's x0 lives on the device; nothing needs it any more.
            self._x = None
        else:
            # The live preview shows the first variation only. Rendering and
            # post-processing the other three for it was the bulk of a 4-variation
            # preview's cost.
            raws, pps = self._rendered(self._x[:1])
            self._first = (raws[0], pps[0])

    def __getitem__(self, key):
        if key in self._meta:
            return self._meta[key]
        if key in self._SINGLE:
            self._ensure(every=False)
            return self._first[self._SINGLE.index(key)]
        if key in self._image_keys():
            self._ensure(every=True)
            return self._all[self._BATCH.index(key)]
        raise KeyError(key)

    def __iter__(self):
        yield from self._meta
        yield from self._image_keys()

    def __len__(self):
        return len(self._meta) + len(self._image_keys())


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
