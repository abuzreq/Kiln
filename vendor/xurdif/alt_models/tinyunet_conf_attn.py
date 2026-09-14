import math
import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# Command-line / metadata attention configuration
# ============================================================

def parse_attn_config(value):
    """
    Convert a command-line attention specification into a dict.

    Examples:
        None
            -> None

        "none"
            -> {}

        "mid:full"
            -> {"mid": "full"}

        "-1:linear,mid:full"
            -> {-1: "linear", "mid": "full"}

        "-2:window,-1:linear,mid:full"
            -> {-2: "window", -1: "linear", "mid": "full"}

    Locations:
        mid = bottleneck
        -1  = last encoder level before bottleneck
        -2  = second-last encoder level
        ...

    Valid attention types:
        full
        linear
        window

    Notes:
        * Keep the original string in model metadata.
        * Use this function only to make the runtime config object.
        * On the command line, prefer:
              --attn=-1:linear,mid:full
          because the value begins with '-'.
    """
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    if value.lower() == "none":
        return {}

    valid_types = {"full", "linear", "window"}
    config = {}

    for item in value.split(","):
        item = item.strip()

        if ":" not in item:
            raise ValueError(
                f"Invalid attention specification '{item}'. "
                "Expected LOCATION:TYPE, for example "
                "'-1:linear' or 'mid:full'."
            )

        location, kind = item.split(":", 1)
        location = location.strip()
        kind = kind.strip().lower()

        if kind not in valid_types:
            raise ValueError(
                f"Unknown attention type '{kind}'. "
                f"Valid types are: {', '.join(sorted(valid_types))}"
            )

        if location.lower() == "mid":
            key = "mid"
        else:
            try:
                key = int(location)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid attention location '{location}'. "
                    "Use 'mid' or a negative integer such as -1 or -2."
                ) from exc

            if key >= 0:
                raise ValueError(
                    f"Invalid attention location '{location}'. "
                    "Encoder locations must be negative: "
                    "-1 is closest to the bottleneck."
                )

        if key in config:
            raise ValueError(
                f"Attention location '{location}' was specified more than once."
            )

        config[key] = kind

    return config


# ============================================================
# Helper modules
# ============================================================

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt()


class FiLM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_dim * 2)
        )

    def forward(self, x, t):
        gamma, beta = self.mlp(t).chunk(2, dim=1)
        return (
            x * (1 + gamma[:, :, None, None])
            + beta[:, :, None, None]
        )


class ConvBlock(nn.Module):
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
        emb = torch.exp(
            torch.arange(half, device=x.device) * -emb
        )
        emb = x[:, None].float() * emb[None, :]
        return torch.cat(
            [torch.sin(emb), torch.cos(emb)],
            dim=-1
        )


# ============================================================
# Attention implementations
# ============================================================

class FullAttention2d(nn.Module):
    """
    Full spatial self-attention using PyTorch SDPA.

    PyTorch can choose a memory-efficient / Flash Attention backend
    where supported by the device, dtype and tensor layout.
    """

    def __init__(self, channels, heads=4):
        super().__init__()

        if channels % heads != 0:
            heads = 1

        self.heads = heads
        self.head_dim = channels // heads

        self.norm = LayerNorm()
        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1
        )
        self.proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

    def forward(self, x):
        residual = x

        x = self.norm(x)

        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).chunk(3, dim=1)

        # B,C,H,W -> B,heads,N,head_dim
        q = q.reshape(
            B, self.heads, self.head_dim, N
        ).transpose(-2, -1)

        k = k.reshape(
            B, self.heads, self.head_dim, N
        ).transpose(-2, -1)

        v = v.reshape(
            B, self.heads, self.head_dim, N
        ).transpose(-2, -1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False
        )

        # B,heads,N,head_dim -> B,C,H,W
        out = out.transpose(-2, -1)
        out = out.reshape(B, C, H, W)

        return residual + self.proj(out)


class LinearAttention2d(nn.Module):
    """
    Memory-efficient linear attention.

    No N x N spatial attention matrix is constructed.

    q is normalized over feature/channel dimension.
    k is normalized over spatial positions.
    A small head_dim x head_dim context matrix is formed first.
    """

    def __init__(
        self,
        channels,
        heads=4,
        head_dim=32
    ):
        super().__init__()

        self.heads = heads
        self.head_dim = head_dim

        hidden = heads * head_dim

        self.norm = LayerNorm()

        self.to_qkv = nn.Conv2d(
            channels,
            hidden * 3,
            kernel_size=1,
            bias=False
        )

        self.to_out = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1
        )

    def forward(self, x):
        residual = x

        x = self.norm(x)

        B, _, H, W = x.shape
        N = H * W

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        # B,(heads*head_dim),H,W
        # -> B,heads,head_dim,N
        q = q.reshape(
            B, self.heads, self.head_dim, N
        )

        k = k.reshape(
            B, self.heads, self.head_dim, N
        )

        v = v.reshape(
            B, self.heads, self.head_dim, N
        )

        # Standard linear-attention factorisation:
        # q normalized over features,
        # k normalized over positions.
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        # B,h,d,N @ B,h,N,d -> B,h,d,d
        context = torch.matmul(
            k,
            v.transpose(-1, -2)
        )

        # B,h,d,d @ B,h,d,N -> B,h,d,N
        out = torch.matmul(
            context,
            q
        )

        out = out.reshape(
            B,
            self.heads * self.head_dim,
            H,
            W
        )

        return residual + self.to_out(out)


class WindowAttention2d(nn.Module):
    """
    Full attention independently inside fixed spatial windows.

    This keeps attention local and avoids a full-image N x N matrix.
    """

    def __init__(
        self,
        channels,
        heads=4,
        window_size=8
    ):
        super().__init__()

        if channels % heads != 0:
            heads = 1

        self.heads = heads
        self.head_dim = channels // heads
        self.window_size = window_size

        self.norm = LayerNorm()

        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1
        )

        self.proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

    def _make_windows(self, t, B, C, Hp, Wp, ws):
        """
        B,C,Hp,Wp -> windows,heads,tokens,head_dim
        """
        t = t.reshape(
            B,
            C,
            Hp // ws,
            ws,
            Wp // ws,
            ws
        )

        t = t.permute(
            0, 2, 4, 1, 3, 5
        )

        t = t.reshape(
            -1,
            C,
            ws * ws
        )

        t = t.reshape(
            -1,
            self.heads,
            self.head_dim,
            ws * ws
        ).transpose(-2, -1)

        return t

    def forward(self, x):
        residual = x

        x = self.norm(x)

        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h or pad_w:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h)
            )

        _, _, Hp, Wp = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = self._make_windows(q, B, C, Hp, Wp, ws)
        k = self._make_windows(k, B, C, Hp, Wp, ws)
        v = self._make_windows(v, B, C, Hp, Wp, ws)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False
        )

        # windows,heads,tokens,head_dim
        # -> windows,C,ws,ws
        out = out.transpose(-2, -1)
        out = out.reshape(
            -1,
            C,
            ws,
            ws
        )

        nH = Hp // ws
        nW = Wp // ws

        # reconstruct the padded feature map
        out = out.reshape(
            B,
            nH,
            nW,
            C,
            ws,
            ws
        )

        out = out.permute(
            0, 3, 1, 4, 2, 5
        )

        out = out.reshape(
            B,
            C,
            Hp,
            Wp
        )

        # remove any padding
        out = out[:, :, :H, :W]

        return residual + self.proj(out)


def make_attention(
    kind,
    channels,
    heads=4,
    window_size=8,
    linear_head_dim=32
):
    """
    Build one attention module.

    kind:
        None / "none" -> Identity
        "full"        -> FullAttention2d
        "linear"      -> LinearAttention2d
        "window"      -> WindowAttention2d
    """
    if kind is None or kind == "none":
        return nn.Identity()

    if kind == "full":
        return FullAttention2d(
            channels,
            heads=heads
        )

    if kind == "linear":
        return LinearAttention2d(
            channels,
            heads=heads,
            head_dim=linear_head_dim
        )

    if kind == "window":
        return WindowAttention2d(
            channels,
            heads=heads,
            window_size=window_size
        )

    raise ValueError(
        f"Unknown attention type: {kind}"
    )


# ============================================================
# TinyUNet
# ============================================================

class TinyUNetWithAttn(nn.Module):
    """
    Tiny U-Net with configurable attention on the down path
    and at the bottleneck.

    Example runtime attn_config values:

        None
            historical/default behaviour:
            full attention at bottleneck

        {}
            no attention

        {"mid": "full"}
            full bottleneck attention

        {-1: "linear", "mid": "full"}
            linear attention one scale above bottleneck,
            full attention at bottleneck

        {-2: "window", -1: "linear", "mid": "full"}
            window attention two scales above bottleneck,
            linear attention one scale above,
            full attention at bottleneck

    Negative encoder locations are relative to the bottleneck:

        -1 = last encoder block
        -2 = second-last encoder block
        ...
    """

    def __init__(
        self,
        dim=64,
        channels=3,
        dim_mults=(1, 2, 4),
        out_dim=None,
        attn_config=None,
        attn_heads=4,
        window_size=8,
        linear_head_dim=32
    ):
        super().__init__()

        # Backward-compatible default:
        # the old model had full attention only at the bottleneck.
        if attn_config is None:
            attn_config = {
                "mid": "full"
            }

        self.init_conv = nn.Conv2d(
            channels,
            dim,
            3,
            padding=1
        )

        dims = [
            dim,
            *map(lambda m: dim * m, dim_mults)
        ]

        in_out = list(
            zip(dims[:-1], dims[1:])
        )

        n_levels = len(in_out)

        # ----------------------------------------------------
        # Convert relative negative locations to internal
        # zero-based encoder indices.
        #
        # With 5 levels:
        #     -5 -> 0
        #     -4 -> 1
        #     -3 -> 2
        #     -2 -> 3
        #     -1 -> 4
        # ----------------------------------------------------

        down_attn_config = {}

        for location, kind in attn_config.items():

            if location == "mid":
                continue

            if not isinstance(location, int) or location >= 0:
                raise ValueError(
                    f"Invalid attention location: {location}. "
                    "Use 'mid' or a negative encoder level."
                )

            level = n_levels + location

            if level < 0 or level >= n_levels:
                raise ValueError(
                    f"Attention level {location} is outside the network. "
                    f"With {n_levels} encoder levels, valid locations are "
                    f"-1 through -{n_levels}, plus 'mid'."
                )

            down_attn_config[level] = kind

        # ----------------------------
        # Time embedding
        # ----------------------------

        time_dim = dim * 4

        self.time_emb = SinusoidalPosEmb(dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # ----------------------------
        # Down path
        # ----------------------------

        self.downs = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.skip_dims = []

        for level, (dim_in, dim_out) in enumerate(in_out):

            block = ConvBlock(
                dim_in,
                dim_out,
                time_emb_dim=time_dim
            )

            down = nn.Conv2d(
                dim_out,
                dim_out,
                4,
                stride=2,
                padding=1
            )

            kind = down_attn_config.get(
                level,
                None
            )

            attn = make_attention(
                kind,
                dim_out,
                heads=attn_heads,
                window_size=window_size,
                linear_head_dim=linear_head_dim
            )

            self.downs.append(
                nn.ModuleList([
                    block,
                    down
                ])
            )

            self.down_attns.append(attn)
            self.skip_dims.append(dim_out)

        # ----------------------------
        # Bottleneck
        # ----------------------------

        mid_dim = dims[-1]

        self.mid_block1 = ConvBlock(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim
        )

        self.mid_attn = make_attention(
            attn_config.get("mid", None),
            mid_dim,
            heads=attn_heads,
            window_size=window_size,
            linear_head_dim=linear_head_dim
        )

        self.mid_block2 = ConvBlock(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim
        )

        # ----------------------------
        # Up path
        # ----------------------------

        self.ups = nn.ModuleList()

        for dim_out, skip_dim in zip(
            reversed(dims[:-1]),
            reversed(self.skip_dims)
        ):

            up = nn.ConvTranspose2d(
                mid_dim,
                dim_out,
                4,
                stride=2,
                padding=1
            )

            block = ConvBlock(
                dim_out + skip_dim,
                dim_out,
                time_emb_dim=time_dim
            )

            self.ups.append(
                nn.ModuleList([
                    up,
                    block
                ])
            )

            mid_dim = dim_out

        self.final_conv = nn.Conv2d(
            dim,
            out_dim or channels,
            1
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(
            self.time_emb(time)
        )

        skips = []

        for (block, down), attn in zip(
            self.downs,
            self.down_attns
        ):
            x = block(x, t)

            # Optional attention at this resolution,
            # before downsampling.
            x = attn(x)

            skips.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for (up, block), skip in zip(
            self.ups,
            reversed(skips)
        ):
            x = up(x)

            x = torch.cat(
                (x, skip),
                dim=1
            )

            x = block(x, t)

        return self.final_conv(x)
