import torch
import torch.nn.functional as F
from math import ceil
from kernel_funcs import cubic, KernelMLP
EPS = torch.finfo().tiny


def resize(input, scale_factors=None, out_sz=None, kernel_func=cubic, kernel_shape=(4, 4)):
    # set flexibility for inputs
    scale_factors, in_sz, out_sz, kernel_shape =  set_scale_and_out_sz(input, 
                                                                       out_sz, 
                                                                       scale_factors, 
                                                                       kernel_shape)
    
    # get projected grid- sub-pixel source locations on the input for each output pixel
    projected_grid = get_grid(scale_factors, 
                              in_sz, 
                              out_sz, 
                              input.dtype, 
                              input.device)

    # pad input and fix projected grid accordingly
    input_padded, projected_grid = pad(input, 
                                       projected_grid, 
                                       kernel_shape)
    
    # for each output pixel, get neighbors and distances from the input pixels
    neighbors, dists = get_neighbors(input_padded,
                                     projected_grid,
                                     kernel_shape)
    
    # calculate weights for each neighbor for each output pixel
    weights = kernel_func(dists, scale_factors)
    
    # apply weights to neighbors to calculate output pixel values
    output = (weights * neighbors).sum(1)

    return output.view(*input.shape[:2], *out_sz)



@torch.no_grad()
def get_grid_1d(scale, in_size, out_size, dtype, device):
    out_coords = torch.arange(out_size, dtype=dtype, device=device)
    # shift to the center
    out_coords = out_coords - (out_size - in_size * scale) / 2
    out_grid = out_coords / scale + 0.5 * (1.0 / scale - 1.0)
    return out_grid


@torch.no_grad()
def get_grid(scale, in_size, out_size, dtype, device):
    grids = [get_grid_1d(s, insz, outsz, dtype, device)
                for s, insz, outsz in zip(scale, in_size, out_size)]
    grid = torch.stack(torch.meshgrid(*grids), 0)
    return grid


def get_neighbors(input, grid, kernel_shape):
    with torch.no_grad():
        win_coords_1d = [torch.arange(sz.item(), dtype=grid.dtype, device=grid.device) 
                         for sz in kernel_shape]
        raw_dists = torch.stack(torch.meshgrid(win_coords_1d), 0).view(2, -1, 1, 1)
        num_neighbors = len(win_coords_1d[0]) * len(win_coords_1d[1])

        top_left_float = grid - kernel_shape[:, None, None] / 2
        top_left = torch.ceil(top_left_float - EPS)
        neighbors_coords = (top_left.unsqueeze(1) + raw_dists).long()  # [2, num_neighbors, H, W]
        dists = neighbors_coords - grid.unsqueeze(1)
        if input is None:
            return dists
        
    neighbors = input[:, :, neighbors_coords[0], neighbors_coords[1]]
    neighbors = neighbors.view(input.shape[0]*input.shape[1], num_neighbors, *grid.shape[-2:])
    return neighbors, dists


def pad(input, grid, kernel_shape):
    with torch.no_grad():
        pad_size = -torch.ceil(grid[:, 0, 0] - kernel_shape / 2)
        pad = (int(pad_size[0]), int(pad_size[1]))
        padding = (pad[1], pad[1], pad[0], pad[0])
        grid += pad_size[:, None, None]
    padded = F.pad(input, padding)
    return padded, grid


def set_scale_and_out_sz(input, out_shape, scale_factors, kernel_shape):
    # eventually we must have both scale-factors and out-sizes for all in/out
    # dims. however, we support many possible partial arguments
    if scale_factors is None and out_shape is None:
        raise ValueError("either scale_factors or out_shape should be "
                         "provided")
    in_shape = input.shape[-2:]
    if out_shape is not None:
        # if out_shape has less dims than in_shape, we defaultly resize the
        # first dims for numpy and last dims for torch
        out_shape = list(in_shape[:-len(out_shape)]) + list(out_shape)
        if scale_factors is None:
            # if no scale given, we calculate it as the out to in ratio
            # (not recomended)
            scale_factors = [out_sz / in_sz for out_sz, in_sz
                             in zip(out_shape, in_shape)]
    if scale_factors is not None:
        # by default, if a single number is given as scale, we assume resizing
        # two dims (most common are images with 2 spatial dims)
        scale_factors = (scale_factors
                         if isinstance(scale_factors, (list, tuple))
                         else [scale_factors, scale_factors])
        # if less scale_factors than in_shape dims, we defaultly resize the
        # first dims for numpy and last dims for torch
        scale_factors = ([1] * (len(in_shape) - len(scale_factors)) +
                         list(scale_factors))
        if out_shape is None:
            # when no out_shape given, it is calculated by multiplying the
            # scale by the in_shape (not recomended)
            out_shape = [ceil(scale_factor * in_sz)
                         for scale_factor, in_sz in
                         zip(scale_factors, in_shape)]
            
    # handle kernel_shape
    kernel_shape = [kernel_shape]*2 if not isinstance(kernel_shape, (list, tuple)) else kernel_shape
    kernel_shape = torch.tensor(kernel_shape, device=input.device)

    return scale_factors, in_shape, out_shape, kernel_shape


def one_k_opt_step(high_res, low_res, k_net, k_shape, opt, sf=None):
    sf = (high_res.shape[-2] / low_res.shape[-2], 
          high_res.shape[-1] / low_res.shape[-1]) if sf is None else sf
    low_res_pred = resize(high_res, (1/sf).tolist(), kernel_func=k_net, kernel_shape=k_shape)
    loss = (low_res_pred - low_res.detach()).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss
