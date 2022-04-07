import torch.nn as nn
import torch
from collections import OrderedDict

class Generator(nn.Module):
    '''
    Simple 2-layer MLP encoder network
    '''
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = [('fc1', nn.Linear(in_dim, latent_dim[0])), ('relu1', nn.LeakyReLU(0.2))]
        # self.fc1 = [('bn1', nn.BatchNorm1d(in_dim)), ('fc1', nn.Linear(in_dim, latent_dim[0])), ('relu1', nn.LeakyReLU(0.2))]
        self.fcs = []
        for i in range(1, len(latent_dim)):
            # self.fcs.append((f'bn{i+1}', nn.BatchNorm1d(latent_dim[i-1])))
            self.fcs.append((f'fc{i+1}', nn.Linear(latent_dim[i-1], latent_dim[i])))
            self.fcs.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.fc2 = [# (f'bn{len(latent_dim) + 1}', nn.BatchNorm1d(latent_dim[-1])), 
                    (f'fc{len(latent_dim) + 1}', nn.Linear(latent_dim[-1], out_dim))]
        self.layers = nn.Sequential(OrderedDict(self.fc1 + self.fcs + self.fc2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    '''
    Simple 2-layer MLP decoder network
    '''
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = [('fc1', nn.Linear(in_dim, latent_dim[0])), ('sig1', nn.Sigmoid())]
        # self.fc1 = [('bn1', nn.BatchNorm1d(in_dim)), ('fc1', nn.Linear(in_dim, latent_dim[0])), ('sig1', nn.Sigmoid())]
        self.fcs = []
        for i in range(1, len(latent_dim)):
            # self.fcs.append((f'bn{i+1}', nn.BatchNorm1d(latent_dim[i-1])))
            self.fcs.append((f'fc{i+1}', nn.Linear(latent_dim[i-1], latent_dim[i])))
            self.fcs.append((f'sig{i+1}', nn.Sigmoid()))
        self.fc2 = [ #(f'bn{len(latent_dim) + 1}', nn.BatchNorm1d(latent_dim[-1])), 
                    (f'fc{len(latent_dim) + 1}', nn.Linear(latent_dim[-1], out_dim)),
                    (f'sig{len(latent_dim) + 1}', nn.Sigmoid())]
        self.layers = nn.Sequential(OrderedDict(self.fc1 + self.fcs + self.fc2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
