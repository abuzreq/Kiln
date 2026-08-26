"""xurdif's training objective, ported out of the vendored engine.

Kiln's compact models are not trained the way a stock DDPM is. Standard practice
is MSE on the predicted noise; ``GaussianDiffusion.p_losses`` instead works in
**x0 space** and optimises

    l1w * edge_weighted_l1(pred_x0, target) + ssimw * (1 - SSIM(pred_x0, target))

where the L1 term multiplies error on the *target's* Sobel edges by ``1 + 4``.
That edge weighting is most of why these models hold line and texture character
on small datasets, so a Diffusers training loop that only offers MSE is not a
replacement for the xurdif trainer -- it is a different trainer.

As with ``tinyunet.py``, the functions here are an op-for-op port, not an
improvement, and ``scripts/smoke_tinyunet_parity.py`` diffs them against the
vendored originals.

One faithfully-reproduced quirk: the vendored code calls ``pytorch_msssim.ssim``
on data already scaled to [0,1] but leaves ``data_range`` at its default of
**255**. That inflates SSIM's stabilising constants by ~65000x, which pushes the
structural term close to zero and makes it far weaker than its weight suggests.
It is reproduced rather than corrected because every existing model was trained
this way; Kiln's default ``ssimw`` is 0, so it is usually inert regardless.
"""
import torch
import torch.nn.functional as F

# The values GaussianDiffusion.p_losses passes, which differ from
# edge_weighted_l1's own defaults (0.1) -- the call site wins.
EDGE_WEIGHT = 4.0
EDGE_THRESHOLD = 0.08
# The vendored call omits data_range, so pytorch_msssim's default applies.
SSIM_DATA_RANGE = 255


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Edge magnitude of ``x`` (B,C,H,W) as (B,1,H,W)."""
    x_gray = x.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor([[1., 0., -1.],
                            [2., 0., -2.],
                            [1., 0., -1.]], device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1., 2., 1.],
                            [0., 0., 0.],
                            [-1., -2., -1.]], device=x.device).view(1, 1, 3, 3)

    gx = F.conv2d(x_gray, sobel_x, padding=1)
    gy = F.conv2d(x_gray, sobel_y, padding=1)

    edges = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return edges


def edge_weighted_l1(x_pred: torch.Tensor, x_target: torch.Tensor,
                     edge_weight: float = EDGE_WEIGHT,
                     edge_threshold: float = 0.1):
    """L1 with error on the target's edge pixels up-weighted.

    Returns ``(loss, parts)``, matching the vendored signature so the two can be
    compared directly.
    """
    base_l1 = F.l1_loss(x_pred, x_target)

    edges = sobel_edges(x_target)
    edge_mask = (edges > edge_threshold).float()

    diff = torch.abs(x_pred - x_target).mean(dim=1, keepdim=True)
    weighted_diff = diff * (1.0 + edge_weight * edge_mask)
    edge_l1 = weighted_diff.mean()

    loss = edge_l1
    return loss, {"base_l1": base_l1.detach(), "edge_l1": edge_l1.detach()}


def predict_x0_from_eps(x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
                        alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """x0 from a noise prediction, the way the vendored engine does it."""
    ac = alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[t]
    shape = (-1,) + (1,) * (x_t.dim() - 1)
    sqrt_recip = torch.sqrt(1.0 / ac).reshape(shape)
    sqrt_recipm1 = torch.sqrt(1.0 / ac - 1).reshape(shape)
    return sqrt_recip * x_t - sqrt_recipm1 * eps


def xurdif_objective(pred_x0: torch.Tensor, x_start: torch.Tensor,
                     l1w: float = 1.0, ssimw: float = 0.0,
                     edge_weight: float = EDGE_WEIGHT,
                     edge_threshold: float = EDGE_THRESHOLD):
    """The composed loss from ``GaussianDiffusion.p_losses``.

    Returns ``(loss, parts)``. ``x_start`` is detached: no gradient flows into
    the data, matching the vendored behaviour.
    """
    from pytorch_msssim import ssim

    pred_x0 = pred_x0.clamp(-1, 1)            # keep grads
    x_tgt = x_start.clamp(-1, 1).detach()     # no grads into data

    loss_x0_l1, parts = edge_weighted_l1(
        x_pred=pred_x0, x_target=x_tgt,
        edge_weight=edge_weight, edge_threshold=edge_threshold,
    )

    pred01 = (pred_x0 * 0.5 + 0.5).clamp(0, 1)
    tgt01 = (x_tgt * 0.5 + 0.5)
    loss_ssim = 1.0 - ssim(pred01, tgt01)

    loss = l1w * loss_x0_l1 + ssimw * loss_ssim
    return loss, {**parts, "ssim": loss_ssim.detach(), "l1": loss_x0_l1.detach()}


def compute_loss(model_out: torch.Tensor, noisy: torch.Tensor, noise: torch.Tensor,
                 clean: torch.Tensor, t: torch.Tensor, alphas_cumprod: torch.Tensor,
                 prediction_type: str = "epsilon", objective: str = "mse",
                 l1w: float = 1.0, ssimw: float = 0.0):
    """One training loss, for either objective.

    ``mse`` is the stock Diffusers target (noise for epsilon-prediction, the
    clean image for sample-prediction). ``xurdif`` converts to x0 space first --
    which it must, because the edge weighting is only meaningful on an image.
    """
    if objective == "mse":
        target = noise if prediction_type == "epsilon" else clean
        return F.mse_loss(model_out.float(), target.float()), {}

    if objective != "xurdif":
        raise ValueError(f"unknown objective: {objective}")

    if prediction_type == "epsilon":
        pred_x0 = predict_x0_from_eps(noisy, t, model_out, alphas_cumprod)
    else:
        pred_x0 = model_out
    return xurdif_objective(pred_x0, clean, l1w=l1w, ssimw=ssimw)
