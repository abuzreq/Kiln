"""The xurdif TinyUNet architecture, re-homed as a first-class Diffusers model.

Why this exists: the compact network is worth keeping on its own merits -- at
512px it is ~4x faster to sample and ~7x faster per training step than the
closest ``UNet2DModel`` config, on a fraction of the training memory -- but the
vendored engine around it (a CUDA-only subprocess trainer, a bespoke ``.pt``
pickle format, two gradio apps) is not. Re-homing separates the two: the same
network, in a format the ecosystem understands, with ``save_pretrained``,
safetensors, PEFT and Accelerate for free.

**The module definitions below are a deliberate op-for-op copy of
``vendor/xurdif/alt_models/tinyunet_with_attn3.py``.** They are not an
improvement, a tidy-up, or a re-derivation: every existing Kiln checkpoint was
trained against these exact operations in this exact order, so any change --
including one that looks numerically harmless -- would silently alter what those
models produce. ``scripts/smoke_tinyunet_parity.py`` asserts bit-identical
forwards, gradients and training steps against the vendored class, and is the
guard against drift. If it fails, this file is wrong, not the test.

Two things are load-bearing and easy to break by accident:

- **Class names.** Craft keys its layer types and its "attention"/"blocks" group
  chips off ``type(module).__name__`` (see ``backends/xurdif/graph.py``), so
  ``ConvBlock`` and ``SelfAttention2d`` must keep those names.
- **Attribute names and nesting.** The state dict keys are the module paths, and
  they must match the vendored network exactly so existing ``.pt`` weights load
  without remapping. Verified: ``ModelMixin`` adds nothing to a state dict.
"""
import math

import torch
import torch.nn as nn
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.unets.unet_2d import UNet2DOutput

# --- Helper modules: verbatim from the vendored architecture ---------------


class LayerNorm(nn.Module):
    """Channel-wise normalisation with no learnable affine.

    Not ``nn.LayerNorm``: it normalises over dim=1 (channels) per spatial
    location and has zero parameters -- the scale and shift come from FiLM
    instead. Substituting a standard norm here changes every trained model.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt()


class FiLM(nn.Module):
    """Time conditioning as scale *and* shift.

    Richer than the shift-only conditioning ``ResnetBlock2D`` uses by default.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_dim * 2)
        )

    def forward(self, x, t):
        gamma, beta = self.mlp(t).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ConvBlock(nn.Module):
    """One 3x3 conv, normed, FiLM-conditioned, activated. No residual path.

    The lack of a residual is the defining difference from ``ResnetBlock2D``
    (which runs two convs plus a shortcut) and is most of why this network is so
    much cheaper per level.

    Attribute definition order is preserved from the vendored class so
    ``named_modules()`` iterates identically.
    """

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = LayerNorm()
        self.act = nn.SiLU()
        self.film = FiLM(time_emb_dim, out_ch)

    def forward(self, x, t):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, t)
        return self.act(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class SelfAttention2d(nn.Module):
    """Single-head attention over the full feature map.

    Two details differ from ``diffusers.models.attention_processor.Attention``
    and both are deliberate: the scale is ``1/sqrt(C)`` rather than
    ``1/sqrt(head_dim)``, and the residual is added *before* the output
    projection (``proj(out + x)``, not ``x + proj(out)``).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, -1).permute(0, 2, 1)
        k = self.k(x).reshape(B, C, -1)
        v = self.v(x).reshape(B, C, -1).permute(0, 2, 1)
        attn = torch.bmm(q, k) / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj(out + x)


# --- The model ------------------------------------------------------------

# The vendored ``mtype`` this class reproduces. Recorded in the converted
# model's config so a repo can be traced back to the architecture it came from.
SOURCE_MTYPE = "tinyunet_with_attention3"


class TinyUNet2DModel(ModelMixin, ConfigMixin):
    """xurdif's TinyUNetWithAttn with a Diffusers surface.

    The body is the vendored network unchanged; what is added is the config
    registration and the ``(sample, timestep) -> UNet2DOutput`` calling
    convention the rest of the ecosystem expects.
    """

    @register_to_config
    def __init__(self, dim: int = 64, channels: int = 3,
                 dim_mults=(1, 2, 2, 2), out_dim=None,
                 sample_size=None, source_mtype: str = SOURCE_MTYPE):
        super().__init__()
        dim_mults = tuple(dim_mults)
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embedding
        time_dim = dim * 4
        self.time_emb = SinusoidalPosEmb(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Downsampling
        self.downs = nn.ModuleList()
        self.skip_dims = []
        for dim_in, dim_out in in_out:
            block = ConvBlock(dim_in, dim_out, time_emb_dim=time_dim)
            down = nn.Conv2d(dim_out, dim_out, 4, stride=2, padding=1)
            self.downs.append(nn.ModuleList([block, down]))
            self.skip_dims.append(dim_out)

        # Mid block
        mid_dim = dims[-1]
        self.mid_block1 = ConvBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = SelfAttention2d(mid_dim)
        self.mid_block2 = ConvBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling (aligned with skip_dims)
        self.ups = nn.ModuleList()
        for dim_out, skip_dim in zip(reversed(dims[:-1]), reversed(self.skip_dims)):
            up = nn.ConvTranspose2d(mid_dim, dim_out, 4, stride=2, padding=1)
            block = ConvBlock(dim_out + skip_dim, dim_out, time_emb_dim=time_dim)
            self.ups.append(nn.ModuleList([up, block]))
            mid_dim = dim_out  # update for next iteration

        self.final_conv = nn.Conv2d(dim, out_dim or channels, 1)

    def denoise(self, x, time):
        """The vendored forward, op for op.

        Kept separate from ``forward`` so the Diffusers calling convention is a
        thin shell over code that can be compared line by line with the vendored
        class.
        """
        x = self.init_conv(x)
        t = self.time_mlp(self.time_emb(time))

        skips = []

        for block, down in self.downs:
            x = block(x, t)
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for (up, block), skip in zip(self.ups, reversed(skips)):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
            x = block(x, t)

        x = self.final_conv(x)
        return x

    def forward(self, sample, timestep, class_labels=None, return_dict: bool = True):
        """``class_labels`` is accepted and ignored.

        This network is unconditional; the argument exists only so the signature
        is drop-in compatible with ``UNet2DModel`` for callers that pass it.
        """
        out = self.denoise(sample, timestep)
        if not return_dict:
            return (out,)
        return UNet2DOutput(sample=out)

    # --- interop ------------------------------------------------------
    @property
    def size_multiple(self) -> int:
        """Dimensions must divide by this for the skip concatenations to line up."""
        return 2 ** len(tuple(self.config.dim_mults))

    @classmethod
    def from_vendored(cls, net, sample_size=None) -> "TinyUNet2DModel":
        """Build from an already-constructed vendored ``TinyUNetWithAttn``.

        Reads the shape back off the network rather than trusting a caller to
        restate it, then transplants the weights verbatim.
        """
        dim = net.init_conv.out_channels
        channels = net.init_conv.in_channels
        dim_mults = tuple(d // dim for d in net.skip_dims)
        out = cls(dim=dim, channels=channels, dim_mults=dim_mults,
                  out_dim=net.final_conv.out_channels, sample_size=sample_size)
        missing, unexpected = out.load_state_dict(net.state_dict(), strict=True)
        return out
