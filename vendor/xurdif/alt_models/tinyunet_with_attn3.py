import torch
from torch import nn
import torch.nn.functional as F
import math

# --- Helper Modules ---

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
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

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
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class SelfAttention2d(nn.Module):
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
# --- TinyUNet ---

class TinyUNetWithAttn(nn.Module):
    def __init__(self, dim=64, channels=3, dim_mults=(1, 2, 4), out_dim=None):
        super().__init__()
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

    def forward(self, x, time):
        input_shape = x.shape
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
