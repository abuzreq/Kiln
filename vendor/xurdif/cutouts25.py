# gpu_cutouts.py
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import math
import torch
import torch.nn.functional as F

@dataclass
class CutoutConfig:
    base_low: int = 32
    base_high: int = 512
    num_min: int = 4
    num_max: int = 64
    strategy: Literal["linear", "loguniform", "mixture"] = "loguniform"
    collapse_range_at_right: bool = True
    margin_frac: float = 0.0   # 0..0.5
    out_size: int = 224
    clip_normalize: bool = True

class GpuCutoutSampler:
    """
    CUDA-friendly, vectorized cutouts for CLIP.
    - Call .sample(img, slider) each step; minimal overhead.
    - img: [B=1, C, H, W] or [C, H, W], on CUDA or CPU; output matches device.
    """
    def __init__(self, cfg: CutoutConfig):
        self.cfg = cfg
        # Pre-store CLIP mean/std on all calls (moved to device at use time)
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
        self._clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)

    @staticmethod
    def _clamp01(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)

    def _derive_params(self, slider: float) -> Tuple[int, int, int]:
        s = float(max(0.0, min(1.0, slider)))
        # num cuts (inverse with s)
        num_cuts = int(round(self.cfg.num_min + (self.cfg.num_max - self.cfg.num_min) * (1.0 - s)))
        num_cuts = max(self.cfg.num_min, min(self.cfg.num_max, num_cuts))
        # low/high
        low = int(round(self.cfg.base_low * (1.0 - s) + self.cfg.base_high * s))
        high = self.cfg.base_high
        if self.cfg.collapse_range_at_right:
            high = int(round(high * (1.0 - s) + low * s))
        if high <= low:
            high = low + 1
        return low, high, num_cuts

    @staticmethod
    def _bands_between(low: int, high: int) -> torch.Tensor:
        # power-of-two-ish band centers between low..high
        if high <= 2:
            return torch.tensor([(low + high)//2], dtype=torch.float32)
        start = max(2, low)
        p2 = 2 ** int(math.floor(math.log2(start)))
        vals = []
        v = p2
        while v < high:
            if low <= v <= high:
                vals.append(float(v))
            v *= 2
        if not vals:
            vals = [float((low + high)/2.0)]
        return torch.tensor(vals, dtype=torch.float32)

    def _sample_sizes(
        self, low: int, high: int, s: float, n: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return [n] integer sizes on device (vectorized)."""
        if self.cfg.strategy == "linear":
            u = torch.rand(n, device=device, dtype=dtype)
            sizes = low + u * (high - low)
        elif self.cfg.strategy == "loguniform":
            lo = math.log(max(1, low))
            hi = math.log(max(low + 1, high))
            u = torch.rand(n, device=device, dtype=dtype)
            sizes = torch.exp(u * (hi - lo) + lo)
        elif self.cfg.strategy == "mixture":
            bands = self._bands_between(low, high).to(device=device, dtype=dtype)
            m = bands.numel()
            if m == 1:
                sizes = bands.expand(n)
            else:
                # weights tilt small→large as s goes 0→1
                xs = torch.linspace(0, 1, steps=m, device=device, dtype=dtype)
                gamma = 0.5 * (1.0 - s) + 2.0 * s
                weights = xs.pow(gamma)
                probs = (weights / (weights.sum() + 1e-12)).clamp_min(1e-12)
                idx = torch.multinomial(probs, num_samples=n, replacement=True)
                chosen = bands[idx]
                # jitter ±25%, clamp to [low, high]
                jitter = (torch.rand(n, device=device, dtype=dtype) * 0.5 + 0.75)
                sizes = (chosen * jitter).clamp(min=float(low), max=float(high))
        else:
            raise ValueError(f"Unknown strategy {self.cfg.strategy}")

        return sizes.round().clamp(min=2).to(torch.int64)

    def _sample_boxes(
        self, H: int, W: int, sizes: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Return boxes [N, 4] where each row is (x0, y0, w, h), integers on device.
        Aspect ratio jitter in [0.8, 1.25], vectorized.
        """
        n = sizes.shape[0]
        mx = int(W * self.cfg.margin_frac)
        my = int(H * self.cfg.margin_frac)
        avail_w = max(2, W - 2*mx)
        avail_h = max(2, H - 2*my)

        # aspect ratio jitter
        ar = 0.8 + torch.rand(n, device=device, dtype=dtype) * (1.25 - 0.8)
        flip = (torch.rand(n, device=device, dtype=dtype) < 0.5)

        w = torch.where(flip, sizes.to(dtype), (sizes * ar).to(dtype))
        h = torch.where(flip, (sizes / ar).to(dtype), sizes.to(dtype))

        w = w.clamp(min=2, max=avail_w).round().to(torch.int64)
        h = h.clamp(min=2, max=avail_h).round().to(torch.int64)

        # positions
        max_x0 = torch.clamp(torch.tensor(W - 2*mx, device=device) - w, min=0)
        max_y0 = torch.clamp(torch.tensor(H - 2*my, device=device) - h, min=0)
        x0 = (torch.rand(n, device=device, dtype=dtype) * (max_x0 + 1).to(dtype)).floor().to(torch.int64) + mx
        y0 = (torch.rand(n, device=device, dtype=dtype) * (max_y0 + 1).to(dtype)).floor().to(torch.int64) + my

        boxes = torch.stack([x0, y0, w, h], dim=-1)  # [N,4]
        return boxes

    @staticmethod
    def _boxes_to_affine(boxes: torch.Tensor, H: int, W: int, out: int) -> torch.Tensor:
        """
        Convert (x0,y0,w,h) to affine matrices for F.grid_sample.
        Returns [N, 2, 3]. Coordinates are in normalized [-1,1].
        """
        # centers in pixels
        x0, y0, w, h = boxes.unbind(-1)
        cx = x0 + w * 0.5
        cy = y0 + h * 0.5

        # scale from src box to target out_size: we want to map a unit square to the box
        # For grid_sample, affine maps target grid -> source coordinates.
        # Construct scaling in normalized coordinates:
        sx = w / (W * 0.5)
        sy = h / (H * 0.5)

        tx = (cx / (W * 0.5)) - 1.0
        ty = (cy / (H * 0.5)) - 1.0

        zeros = torch.zeros_like(sx)
        A = torch.stack([
            torch.stack([sx, zeros, tx], dim=-1),
            torch.stack([zeros, sy, ty], dim=-1),
        ], dim=1)                                # [N, 2, 3]
        return A

    def _extract(self, img: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        Vectorized crop+resize using grid_sample. Returns [N, C, out, out] on img.device.
        """
        added_batch = False
        if img.dim() == 3:
            img = img.unsqueeze(0)
            added_batch = True
        assert img.dim() == 4 and img.size(0) == 1, "Pass a single image [1,C,H,W] or [C,H,W]."

        B, C, H, W = img.shape
        N = boxes.size(0)
        device = img.device
        dtype = img.dtype

        A = self._boxes_to_affine(boxes.to(dtype), H, W, self.cfg.out_size)    # [N,2,3]
        grid = F.affine_grid(A, size=(N, C, self.cfg.out_size, self.cfg.out_size), align_corners=False)  # [N,H,W,2]
        # Expand/Tile image to [N,C,H,W] without copy if possible
        imgN = img.expand(N, -1, -1, -1).contiguous()
        crops = F.grid_sample(imgN, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return crops

    def _maybe_clip_normalize(self, crops: torch.Tensor) -> torch.Tensor:
        if not self.cfg.clip_normalize:
            return crops
        # If in [-1,1], convert to [0,1]
        if crops.min() < 0:
            crops = (crops + 1.0) * 0.5
        mean = self._clip_mean.to(device=crops.device, dtype=crops.dtype)
        std  = self._clip_std.to(device=crops.device, dtype=crops.dtype)
        return (crops - mean) / std

    #@torch.no_grad()
    def sample(self, img: torch.Tensor, slider: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Main entry:
          img: [C,H,W] or [1,C,H,W], float ([-1,1] or [0,1]), any device (CUDA preferred).
          slider: 0..1; 0 = many/small, 1 = few/large.
        Returns:
          crops: [N, C, out_size, out_size] on the same device as img.
        """
        if generator is not None:
            torch.manual_seed(generator.initial_seed())

        if img.dim() == 3:
            _, H, W = img.shape
        else:
            _, _, H, W = img.shape

        device, dtype = img.device, img.dtype
        low, high, num_cuts = self._derive_params(slider)
        sizes = self._sample_sizes(low, high, s=slider, n=num_cuts, device=device, dtype=dtype)
        boxes = self._sample_boxes(H, W, sizes, device=device, dtype=dtype)
        crops = self._extract(img, boxes)
        crops = self._maybe_clip_normalize(crops)
        return crops
