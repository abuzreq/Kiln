# cutouts.py
# Differentiable CLIP cutouts for tensors (no PIL, no .cpu(), no .detach()).
# Expects input in [0,1] range (CLIP-ready after you apply `norm`).

import torch
import torch.nn.functional as F

__all__ = ["cut"]

def _rand_round(x: torch.Tensor) -> torch.Tensor:
    """ stochastic rounding to nearest int (keeps grads wrt input tensor values) """
    # not strictly needed; we use integer crop sizes below. Kept for completeness.
    return (x + torch.rand_like(x)).floor()

def _sample_boxes(B, H, W, cutn, low, high, device):
    # pick a single relative scale s per cutout in [low, high]
    s = torch.empty(cutn, device=device).uniform_(low, high)           # (cutn,)
    h = torch.clamp((s * H).long(), min=1, max=H)                      # (cutn,)
    w = torch.clamp((s * W).long(), min=1, max=W)                      # (cutn,)

    # top-left corners (uniform)
    max_top  = (H - h + 1).clamp(min=1)
    max_left = (W - w + 1).clamp(min=1)
    top  = torch.stack([torch.randint(int(mt.item()), (1,), device=device)[0] for mt in max_top], dim=0)
    left = torch.stack([torch.randint(int(ml.item()), (1,), device=device)[0] for ml in max_left], dim=0)
    return top, left, h, w  # each (cutn,)

def cut(x01, cutn=32, low=0.5, high=1.0, norm=None):
    """
    Args:
        x01:  tensor in [0,1], shape [B,3,H,W], requires_grad=True
        cutn: number of cutouts
        low, high: relative scale range (0..1) of the crop side (w.r.t. H,W)
        norm: optional callable to apply CLIP mean/std (transforms.Normalize)
    Returns:
        Tensor [cutn*B, 3, 224, 224], ready for CLIP.encode_image
    """
    assert x01.is_floating_point()
    assert x01.min() >= 0 - 1e-3 and x01.max() <= 1 + 1e-3, "x01 must be in [0,1] before CLIP norm"
    B, C, H, W = x01.shape
    device = x01.device

    # sample boxes per cutout
    top, left, hh, ww = _sample_boxes(B, H, W, cutn, low, high, device)

    pieces = []
    for i in range(cutn):
        t = int(top[i].item()); l = int(left[i].item())
        h = int(hh[i].item());  w = int(ww[i].item())
        # crop is differentiable w.r.t. pixel values (not positions), which is what we need
        crop = x01[:, :, t:t+h, l:l+w]                        # [B,3,h,w]
        crop = F.interpolate(crop, size=(224, 224), mode="bilinear", align_corners=False)
        if norm is not None:
            crop = norm(crop)                                 # apply CLIP mean/std
        pieces.append(crop)

    return torch.cat(pieces, dim=0)                           # [cutn*B, 3, 224, 224]
