import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class KernelMLP(torch.nn.Module):
    def __init__(self, hidden_layers_chans=None, pos_embed_sz=None, pos_embed_scale=None):
        super(KernelMLP, self).__init__()
        if pos_embed_sz is not None:
            input_chans = 2 + 2 * pos_embed_sz
            self.pos_embed = PosEncoding(pos_embed_sz, pos_embed_scale)
        else:
            input_chans = 4
            self.pos_embed = None

        hidden_layers_chans = [input_chans] + hidden_layers_chans + [1]  # input and output channels
        
        # sequence of linear layers:
        layers = []
        for i in range(len(hidden_layers_chans) - 1):
            layers.append(torch.nn.Linear(hidden_layers_chans[i], 
                                          hidden_layers_chans[i + 1]))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, sf):
        # [2, num_neighbors, H, W] -> [num_neighbors, H, W, 2]
        x = x.permute(1, 2, 3, 0)

        # use scale factors as additional input
        sf_tensor = torch.tensor(sf, dtype=x.dtype, device=x.device)
        sf_tensor = sf_tensor[None, None, None, :].expand(x.shape[0], x.shape[1], x.shape[2], 2)
        x = x * sf_tensor

        # positional encoding
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        x = torch.cat([x, sf_tensor], dim=-1)

        # MLP layers
        for ind, layer in enumerate(self.layers):
            x = layer(x).relu()

        # make sure sum is 1 over neighbors:
        x = x / x.sum(0, keepdim=True)

        # [num_neighbors, H, W, 2] -> [1, num_neighbors, H, W]
        x = x.permute(3, 0, 1, 2)
        return x


def cubic(x):
    # input should be [2, num_neighbors, ...]
    x = x.pow(2).sum(0, keepdim=True).sqrt()  # [1, num_neighbors, ...]
    dtype = x.dtype
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    weights = ((1.5 * absx3 - 2.5 * absx2 + 1.) * (absx <= 1.).to(dtype)
               + (-0.5 * absx3 + 2.5 * absx2 - 4. * absx + 2.) * 
               ((1. < absx) & (absx <= 2.)).to(dtype))
    return weights


def cubic_aa(x, sf):
    sf_tensor = torch.tensor(sf, dtype=x.dtype, device=x.device)
    sf_tensor = sf_tensor[:, None, None, None].expand_as(x)
    weights = cubic(x * sf_tensor) #* math.sqrt(sf[0] * sf[1])
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights


class PosEncoding(nn.Module):
    def __init__(self, embed_sz, scale):
        super().__init__() 
        
        self.b = nn.Parameter(torch.randn(2, embed_sz) * scale, requires_grad=False)
        
    def forward(self, x):
        x_proj = torch.einsum('nhwc,cm->nhwm',(2. * math.pi * x), self.b)
        x_proj = torch.cat([x_proj.sin(), x_proj.cos()], axis=-1)
        return x_proj