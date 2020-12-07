import math
from torch import nn
from torch.autograd import Function
import torch

import gp_interp_cuda

torch.manual_seed(42)


class GPInterpFunction(Function):
    @staticmethod
    def forward(ctx, image, means, stds, radius):
        interp_image, pixel_weights = gp_interp_cuda.forward(
            image.contiguous(),
            means.contiguous(),
            stds.contiguous(),
            radius)
        variables = [image, means, stds, torch.tensor(radius), interp_image, pixel_weights]
        ctx.save_for_backward(*variables)

        return interp_image

    @staticmethod
    def backward(ctx, grad_interp_image):
        image, means, stds, radius, interp_image, pixel_weights = ctx.saved_variables
        radius = radius.item()
        outputs = gp_interp_cuda.backward(
            grad_interp_image.contiguous(),
            image.contiguous(),
            means.contiguous(),
            stds.contiguous(),
            radius,
            interp_image.contiguous(),
            pixel_weights.contiguous(),
        )
        d_image, d_means, d_stds = outputs

        return d_image, d_means, d_stds, None


class GPInterp(nn.Module):
    def __init__(self, h: int, w: int, c: int, downsample_factor: float, std_scale: float, radius: int):
        super(GPInterp, self).__init__()
        # For the initialization, we are going to scatter `num_coords` across `h` x `w` grid
        grid_h = round(h * downsample_factor)
        grid_w = round(w * downsample_factor)
        mean_x = torch.linspace(0, w, grid_w).unsqueeze(0).repeat(grid_h, 1).float() # [grid_h, grid_w]
        mean_y = torch.linspace(0, h, grid_h).unsqueeze(1).repeat(1, grid_w).float() # [grid_h, grid_w]
        means = torch.stack([mean_x, mean_y]).permute(1, 2, 0).view(-1, 2) # [num_coords, 2]
        stds = torch.ones_like(means) * std_scale # [num_coords, 2]

        # Let's break the symmetry in the init
        means = means + torch.rand_like(means) * 0.0001
        stds = stds + torch.rand_like(stds) * 0.0001

        self.means = nn.Parameter(means)
        self.stds = nn.Parameter(stds)
        self.radius = radius

    def forward(self, image):
        return GPInterpFunction.apply(image, self.means, self.stds, self.radius)
