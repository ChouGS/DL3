import numpy as np
import torch
import torch.nn as nn


class SlicedWS(nn.Module):
    def __init__(self, n_slices) -> None:
        super(SlicedWS, self).__init__()
        self.n_slices = n_slices

    def forward(self, x, y):
        SW_loss_val = 0
        proj_angle = torch.rand(self.n_slices) * 2 * torch.pi
        proj_vec = torch.cat([torch.cos(proj_angle).unsqueeze(1), 
                              torch.sin(proj_angle).unsqueeze(1)], 1).unsqueeze(-1)
        x = torch.stack([x.squeeze(0)] * self.n_slices, 0)
        y = torch.stack([y.squeeze(0)] * self.n_slices, 0)
        x_proj = torch.bmm(x, proj_vec).squeeze(2)
        y_proj = torch.bmm(y, proj_vec).squeeze(2)
        x_proj, _ = torch.sort(x_proj, dim=-1)
        y_proj, _ = torch.sort(y_proj, dim=-1)

        SW_loss_val = torch.sum(torch.mean((x_proj - y_proj) ** 2, 1))
        return SW_loss_val
