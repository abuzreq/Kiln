from kornia.enhance import adjust_contrast, adjust_brightness, adjust_saturation, adjust_gamma
from kornia.filters import unsharp_mask, median_blur
from kornia.enhance.equalization import equalize_clahe
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def laplace_sharpen(image: torch.Tensor, blur_radius: int, alpha: float):
        if blur_radius == 0:
            return (image,)
                 
        device = image.device
        batch_size, channels, height, width = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32) * -1
        center = kernel_size // 2
        kernel[center, center] = kernel_size**2
        kernel *= alpha
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1).to(device)
        
        #print(kernel.shape)

        sharpened = F.conv2d(image, kernel, padding=center, groups=channels)
        #sharpened -= sharpened.min()
        #sharpened /= sharpened.max() #- torch.clamp(sharpened, 0, 1)
         
        #result = (1 - alpha) * image + alpha * sharpened 
        result = torch.clamp(sharpened + image, 0, 1)
         
        return (result,)



def to_01(x):      # expects [-1,1]
    return (x + 1) * 0.5

def to_m11(x):     # expects [0,1]
    return x * 2 - 1

def clamp_01(x):
    return x.clamp(0.0, 1.0)

def clamp_m11(x):
    return x.clamp(-1.0, 1.0)

def assert_range(x, lo, hi, name=""):
    # allow tiny float drift
    eps = 1e-3
    mn = float(x.min().detach().cpu())
    mx = float(x.max().detach().cpu())
    if mn < lo - eps or mx > hi + eps:
        raise ValueError(f"{name} out of range [{lo},{hi}] : min={mn:.4f}, max={mx:.4f}")



def pprocess(imT, opt):
    """
    Convention:
      - input imT is in [-1, 1]
      - internal working range is [0, 1]
      - output is in [-1, 1]
    """

    # ---- 0) Convert once to [0,1] and clamp ----
    im01 = clamp_01(to_01(imT))
    # Optional: debug
    # assert_range(im01, 0.0, 1.0, "pprocess.im01(in)")

    # ---- 1) Color/tonal ops (Kornia expects "image-like" floats; keep [0,1]) ----
    if opt.contrast != 1:
        im01 = adjust_contrast(im01, float(opt.contrast))
        im01 = clamp_01(im01)

    if opt.gamma != 1:
        im01 = adjust_gamma(im01, float(opt.gamma))
        im01 = clamp_01(im01)

    if opt.saturation != 1:
        im01 = adjust_saturation(im01, float(opt.saturation))
        im01 = clamp_01(im01)

    # ---- 2) OpenCV filters (must be [0,1] -> uint8 -> [0,1]) ----
    if getattr(opt, "median", 0) > 0:
        img = (im01[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        img = cv2.medianBlur(img, int(opt.median)).astype(np.float32) / 255.0
        im01 = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(imT.device)
        im01 = clamp_01(im01)

    elif opt.bil > 0:
        img = (im01[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        img = cv2.bilateralFilter(img, int(opt.bil), int(opt.bils1), int(opt.bils2)).astype(np.float32) / 255.0
        im01 = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(imT.device)
        im01 = clamp_01(im01)

    # ---- 3) Sharpening (Kornia unsharp_mask; keep [0,1]) ----
    if opt.unsharp > 0:
        im01 = unsharp_mask(
            im01,
            (int(opt.sharpkernel), int(opt.sharpkernel)),
            (float(opt.unsharp), float(opt.unsharp)),
        )
        im01 = clamp_01(im01)

    # ---- 4) Laplace sharpen (your function clamps to [0,1] already) ----
    if hasattr(opt, "laplace_radius") and opt.laplace_radius > 0:
        im01 = laplace_sharpen(im01, int(opt.laplace_radius), float(opt.laplace_alpha))[0]
        im01 = clamp_01(im01)

    # ---- 5) Optional overlay mixing (assume im01 is the base; if you want, preserve a pre-filter copy) ----
    # If you still want the old "ovl0 blend with pre-filter image", do it explicitly:
    # if opt.ovl0 != 0:
    #     im01 = float(opt.ovl0) * im01_pre + (1 - float(opt.ovl0)) * im01
    #     im01 = clamp_01(im01)

    # ---- 6) CLAHE (expects [0,1]) ----
    if opt.eqhist > 0:
        im01 = equalize_clahe(im01, clip_limit=float(opt.eqhist), grid_size=(8, 8))
        im01 = clamp_01(im01)

    # ---- 7) Sharpen-last option ----
    if getattr(opt, "sharpenlast", False) and opt.unsharp > 0:
        im01 = unsharp_mask(
            im01,
            (int(opt.sharpkernel), int(opt.sharpkernel)),
            (float(opt.unsharp), float(opt.unsharp)),
        )
        im01 = clamp_01(im01)

    # ---- 8) Noise (add in [0,1]) ----
    if opt.noise > 0:
        noise = torch.randn_like(im01) * float(opt.noise)
        im01 = clamp_01(im01 + noise)

    # ---- 9) Convert once back to [-1,1] ----
    out = clamp_m11(to_m11(im01))
    return out
