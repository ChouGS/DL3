from collections import OrderedDict
import torch
import torch.nn as nn

class SetNetAttention(nn.Module):
    def __init__(self, latent_dim) -> None:
        super(SetNetAttention, self).__init__()
        self.attention = nn.MultiheadAttention(latent_dim[-1], num_heads=4)
        latent_dim = [1] + latent_dim
        self.q = nn.Sequential(OrderedDict([(f'q{i}', nn.Linear(latent_dim[i-1], latent_dim[i])) for i in range(1, len(latent_dim))]))
        self.k = nn.Sequential(OrderedDict([(f'k{i}', nn.Linear(latent_dim[i-1], latent_dim[i])) for i in range(1, len(latent_dim))]))
        self.v = nn.Sequential(OrderedDict([(f'v{i}', nn.Linear(latent_dim[i-1], latent_dim[i])) for i in range(1, len(latent_dim))]))
        self.outp = nn.Sequential(OrderedDict([(f'outp', nn.Conv1d(latent_dim[-1] * 2, 1, 1))]))

    def forward(self, x):
        # q, k, v as in the attention layer
        q = self.q(x).unsqueeze(1)
        k = self.k(x).unsqueeze(1)
        v = self.v(x).unsqueeze(1)
        
        # Multi-head attention
        att, _ = self.attention(q, k, v)

        # Average and concatenation
        att_mean = torch.mean(att, 0, keepdim=True).repeat(att.shape[0], 1, 1)
        att = torch.cat([att, att_mean], -1).squeeze().transpose(1, 0).unsqueeze(0)

        # 1x1 conv and average again
        outp = torch.mean(self.outp(att))

        return outp
