import numpy as np
import torch
import torch.nn as nn


class SlicedWSLoss2D(nn.Module):
    '''
    Sliced Wasserstein loss for 2D data
    '''
    def __init__(self, n_slices) -> None:
        super(SlicedWSLoss2D, self).__init__()
        self.n_slices = n_slices

    def forward(self, x, y):
        SW_loss_val = 0
        # Projection angle
        proj_angle = torch.rand(self.n_slices) * 2 * torch.pi
        proj_vec = torch.cat([torch.cos(proj_angle).unsqueeze(1), 
                              torch.sin(proj_angle).unsqueeze(1)], 1).unsqueeze(-1)
        x = torch.stack([x.squeeze(0)] * self.n_slices, 0)
        y = torch.stack([y.squeeze(0)] * self.n_slices, 0)

        # Project to 1D
        x_proj = torch.bmm(x, proj_vec).squeeze(2)
        y_proj = torch.bmm(y, proj_vec).squeeze(2)

        # Sort and calculate piecewise SE
        x_proj, _ = torch.sort(x_proj, dim=-1)
        y_proj, _ = torch.sort(y_proj, dim=-1)
        SW_loss_val = torch.sum(torch.mean((x_proj - y_proj) ** 2, 1))

        return SW_loss_val


class SlicedWSLoss3D(nn.Module):
    '''
    Sliced Wasserstein loss for 3D data
    '''
    def __init__(self, n_slices) -> None:
        super(SlicedWSLoss3D, self).__init__()
        self.n_slices = n_slices

    def forward(self, x, y):
        SW_loss_val = 0
        # Projection angle
        proj_a1 = torch.rand(self.n_slices) * 2 * torch.pi
        proj_a2 = torch.rand(self.n_slices) * 2 * torch.pi
        proj_vec = torch.cat([torch.cos(proj_a1).unsqueeze(1), 
                              (torch.sin(proj_a1) * torch.cos(proj_a2)).unsqueeze(1),
                              (torch.sin(proj_a1) * torch.sin(proj_a2)).unsqueeze(1)], 1).unsqueeze(-1)
        x = torch.stack([x.squeeze(0)] * self.n_slices, 0)
        y = torch.stack([y.squeeze(0)] * self.n_slices, 0)
        
        # Project to 1D
        x_proj = torch.bmm(x, proj_vec).squeeze(2)
        y_proj = torch.bmm(y, proj_vec).squeeze(2)
        
        # Sort and calculate piecewise SE
        x_proj, _ = torch.sort(x_proj, dim=-1)
        y_proj, _ = torch.sort(y_proj, dim=-1)
        SW_loss_val = torch.sum(torch.mean((x_proj - y_proj) ** 2, 1))
        
        return SW_loss_val
