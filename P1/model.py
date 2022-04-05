from collections import OrderedDict
from turtle import forward
import torch
import torch.nn as nn

class SetNet(nn.Module):
    def __init__(self, num_layers=9) -> None:
        super(SetNet, self).__init__()
        self.model = torch.nn.Sequential(
            OrderedDict([(f'mp{i}', MessagePassingBlock()) for i in range(num_layers)] + \
                        [('readout', ReadoutBlock())]))
    
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        
        return x

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
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        att, _ = self.attention(q, k, v)
        att_mean = torch.mean(att, 0)
        att_mean = torch.stack([att_mean for _ in range(att.shape[0])], 1).transpose(1, 0)
        att = torch.cat([att, att_mean], 1).transpose(1, 0)
        outp = torch.mean(self.outp(att), 1)
        return outp

class MessagePassingBlock(nn.Module):
    def __init__(self, hidden_dim=64) -> None:
        super(MessagePassingBlock, self).__init__()
        self.message = nn.Conv1d(1, hidden_dim, 7, padding='same')
        self.act = nn.ReLU()
        self.aggregate = nn.Conv1d(2 * hidden_dim, 1, 1, padding='same')

    def forward(self, x):
        '''
        x: 1 x n
        '''
        x_message = self.message(x) # h x n
        aggr_msg = torch.sum(x_message, 1)
        aggr_msg = torch.stack([aggr_msg for _ in range(x_message.shape[1])], 1)
        x = torch.cat([x_message, aggr_msg], 0)
        x = self.aggregate(x)        
        x = self.act(x)
        return x

class ReadoutBlock(nn.Module):
    def __init__(self) -> None:
        super(ReadoutBlock, self).__init__()
    
    def forward(self, x):
        return torch.mean(x)
