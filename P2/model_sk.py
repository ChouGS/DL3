import torch.nn as nn
import torch
from collections import OrderedDict


class Encoder(nn.Module):
    '''
    Simple 2-layer MLP encoder network
    '''
    def __init__(self, in_dim, conv_channels, fc_dims, out_dim) -> None:
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.conv_channels = conv_channels
        self.out_dim = out_dim

        conv_channels = [1] + conv_channels 
        self.convs = []
        for i in range(1, len(conv_channels)):
            self.convs += [(f'conv{i}', nn.Conv1d(conv_channels[i-1], conv_channels[i], 1, padding='same')),
                           (f'relu{i}', nn.LeakyReLU(0.2))]
        self.flatten = [('flatten', nn.Flatten())]
        self.fc = []
        for i in range(1, len(fc_dims)):
            self.fc += [(f'bn{len(conv_channels) + i - 1}', nn.BatchNorm1d(fc_dims[i-1])), 
                        (f'fc{i}', nn.Linear(fc_dims[i-1], fc_dims[i])),
                        (f'relu{len(conv_channels) + i - 1}', nn.LeakyReLU(0.2))]
        self.fc += [(f'bn{len(conv_channels) + len(fc_dims) - 1}', nn.BatchNorm1d(fc_dims[-1])),
                    (f'fc{len(fc_dims)}', nn.Linear(fc_dims[-1], out_dim))]
        self.conv = nn.Sequential(OrderedDict(self.convs + self.flatten))
        self.fc = nn.Sequential(OrderedDict(self.fc))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_dim, fc_dims, conv_channels, out_dim) -> None:
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.conv_channels = conv_channels
        self.out_dim = out_dim

        fc_dims = [in_dim] + fc_dims
        self.fcs = []
        for i in range(1, len(fc_dims)):
            self.fcs += [(f'bn{i}', nn.BatchNorm1d(fc_dims[i-1])),
                         (f'fc{i}', nn.Linear(fc_dims[i-1], fc_dims[i])),
                         (f'relu{i}', nn.LeakyReLU(0.2))]

        self.convs = []
        for i in range(1, len(conv_channels)):
            self.convs += [(f'conv{i}', nn.Conv1d(conv_channels[i-1], conv_channels[i], 1, padding='same')),
                           (f'relu{len(fc_dims) + i - 1}', nn.LeakyReLU(0.2))]
        self.convs += [(f'conv{len(conv_channels)}', nn.Conv1d(conv_channels[-1], 1, 1, padding='same'))]
        
        self.fc = nn.Sequential(OrderedDict(self.fcs))
        self.conv = nn.Sequential(OrderedDict(self.convs))

    def forward(self, x):
        x = self.fc(x)
        x = torch.reshape(x, (-1, self.conv_channels[0], 3))
        for layer in self.conv:
            x = layer(x)
        return x.squeeze()

class SWAutoEncoder(nn.Module):
    '''
    Encoder+Decoder
    '''
    def __init__(self, in_dim, inter_dim) -> None:
        super(SWAutoEncoder, self).__init__()
        self.encoder = Encoder(in_dim, [32, 64, 128, 256], [768, 256, 64], inter_dim)
        self.decoder = Decoder(inter_dim, [64, 256, 768], [256, 128, 64, 32], in_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

    def encode(self, x):
        return self.encoder(x).detach().numpy()

    def decode(self, x):
        # Method for test-only execution
        return self.decoder(x).detach().numpy()


class VEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(VEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.mu = nn.Linear(latent_dim, out_dim)
        self.sigma = nn.Linear(latent_dim, out_dim * out_dim)

    def forward(self, x):
        h = self.fc1(x)
        mu = self.mu(h)
        sigma = self.sigma(h).reshape(-1, self.out_dim, self.out_dim)
        z = torch.randn(x.shape[0], 2)
        return torch.bmm(z.unsqueeze(1), sigma).squeeze() + mu

class VDecoder(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(VDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, out_dim)
        self.sigma = nn.Linear(latent_dim, out_dim * out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_dim, inter_dim) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = VEncoder(in_dim, 32, inter_dim)
        self.decoder = VDecoder(inter_dim, 32, in_dim)
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec
