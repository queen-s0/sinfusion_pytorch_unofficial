import math

import torch
from torch import nn

from einops import rearrange

from sinfusion.convnext import ConvNextBlock
from sinfusion.utils import default


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        channels = 3,
        num_blocks = 8,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding = 3)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        for ind in range(num_blocks):

            self.downs.append(
                ConvNextBlock(dim, dim, 4 * dim, layer_scale_init_value=0)
            )
            self.ups.append(
                ConvNextBlock(dim, dim, 4 * dim, layer_scale_init_value=0)
            )

        self.mid_block = ConvNextBlock(dim, dim, 4 * dim, layer_scale_init_value=0)

        self.out_dim = channels * (1 if not learned_variance else 2)
        
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time=None, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        # r = x.clone()

        t = self.time_mlp(time) if time is None else None

        h = []

        for block in self.downs:
            x = block(x, t)
            h.append(x)

        x = self.mid_block(x, t)

        for block in self.ups:
            # x = torch.cat((x, h.pop()), dim = 1)
            x = x + h.pop()
            x = block(x, t)

        return self.final_conv(x)
