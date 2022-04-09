import torch.nn as nn
import torch
from collections import OrderedDict

class Encoder(nn.Module):
    '''
    Simple 2-layer MLP encoder network
    '''
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = [('bn1', nn.BatchNorm1d(in_dim)), ('fc1', nn.Linear(in_dim, latent_dim[0])), ('relu1', nn.LeakyReLU(0.2))]
        self.fcs = []
        for i in range(1, len(latent_dim)):
            self.fcs.append((f'bn{i+1}', nn.BatchNorm1d(latent_dim[i-1])))
            self.fcs.append((f'fc{i+1}', nn.Linear(latent_dim[i-1], latent_dim[i])))
            self.fcs.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.fc2 = [(f'bn{len(latent_dim) + 1}', nn.BatchNorm1d(latent_dim[-1])), (f'fc{len(latent_dim) + 1}', nn.Linear(latent_dim[-1], out_dim))]
        self.layers = nn.Sequential(OrderedDict(self.fc1 + self.fcs + self.fc2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    '''
    Simple 2-layer MLP decoder network
    '''
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = [('bn1', nn.BatchNorm1d(in_dim)), ('fc1', nn.Linear(in_dim, latent_dim[0])), ('relu1', nn.LeakyReLU(0.2))]
        self.fcs = []
        for i in range(1, len(latent_dim)):
            self.fcs.append((f'bn{i+1}', nn.BatchNorm1d(latent_dim[i-1])))
            self.fcs.append((f'fc{i+1}', nn.Linear(latent_dim[i-1], latent_dim[i])))
            self.fcs.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.fc2 = [(f'bn{len(latent_dim) + 1}', nn.BatchNorm1d(latent_dim[-1])), (f'fc{len(latent_dim) + 1}', nn.Linear(latent_dim[-1], out_dim))]
        self.layers = nn.Sequential(OrderedDict(self.fc1 + self.fcs + self.fc2))

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class SWAutoEncoder(nn.Module):
    '''
    Encoder+Decoder
    '''
    def __init__(self, in_dim, inter_dim) -> None:
        super(SWAutoEncoder, self).__init__()
        self.encoder = Encoder(in_dim, [32, 128, 256, 256, 64], inter_dim)
        self.decoder = Decoder(inter_dim, [64, 256, 256, 128, 32], in_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

    def encode(self, x):
        # Method for test-only execution
        return self.encoder(x).detach().numpy()

    def decode(self, x: torch.Tensor):
        # Method for test-only execution
        return self.decoder(x).detach().numpy()

class VEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(VEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        latent_dim = [in_dim] + latent_dim
        
        # Latent part
        self.fcs = []
        for i in range(1, len(latent_dim) - 1):
            self.fcs.append((f'bn{i}', nn.BatchNorm1d(latent_dim[i-1])))
            self.fcs.append((f'fc{i}', nn.Linear(latent_dim[i-1], latent_dim[i])))
            self.fcs.append((f'relu{i}', nn.LeakyReLU(0.2)))
        self.fc = nn.Sequential(OrderedDict(self.fcs))

        # Mu output branch
        self.mus = [('mu_bn', nn.BatchNorm1d(latent_dim[-2])),
                    ('mu1', nn.Linear(latent_dim[-2], latent_dim[-1])),
                    ('mu_relu', nn.LeakyReLU(0.2)),
                    ('mu2', nn.Linear(latent_dim[-1], out_dim))]
        self.mu = nn.Sequential(OrderedDict(self.mus))

        # Sigma output branch
        self.sigmas = [('sigma_bn', nn.BatchNorm1d(latent_dim[-2])),
                       ('sigma1', nn.Linear(latent_dim[-2], latent_dim[-1])),
                       ('sigma_relu', nn.LeakyReLU(0.2)),
                       ('sigma2', nn.Linear(latent_dim[-1], out_dim))]
        self.sigma = nn.Sequential(OrderedDict(self.sigmas))

    def forward(self, x):
        h = self.fc(x)
        mu = self.mu(h)
        sigma = self.sigma(h)
        return mu, sigma

    def reparametrize(self, bsize, mu, sigma):
        # Generate normal distribution using learned mu and sigma
        z = torch.randn(bsize, self.out_dim)
        return z * torch.exp(0.5 * sigma) + mu

class VDecoder(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(VDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        latent_dim = [in_dim] + latent_dim

        # Latent part
        self.fcs = []
        for i in range(1, len(latent_dim) - 1):
            self.fcs.append((f'bn{i}', nn.BatchNorm1d(latent_dim[i-1])))
            self.fcs.append((f'fc{i}', nn.Linear(latent_dim[i-1], latent_dim[i])))
            self.fcs.append((f'relu{i}', nn.LeakyReLU(0.2)))
        self.fc = nn.Sequential(OrderedDict(self.fcs))

        # Mu output branch
        self.mus = [('mu_bn', nn.BatchNorm1d(latent_dim[-2])),
                    ('mu1', nn.Linear(latent_dim[-2], latent_dim[-1])),
                    ('mu_relu', nn.LeakyReLU(0.2)),
                    ('mu2', nn.Linear(latent_dim[-1], out_dim))]
        self.mu = nn.Sequential(OrderedDict(self.mus))

        # Sigma output branch
        self.sigmas = [('sigma_bn', nn.BatchNorm1d(latent_dim[-2])),
                       ('sigma1', nn.Linear(latent_dim[-2], latent_dim[-1])),
                       ('sigma_relu', nn.LeakyReLU(0.2)),
                       ('sigma2', nn.Linear(latent_dim[-1], out_dim))]
        self.sigma = nn.Sequential(OrderedDict(self.sigmas))

    def forward(self, x):
        h = self.fc(x)
        mu = self.mu(h)
        sigma = self.sigma(h)
        return mu, sigma

    def reparametrize(self, bsize, mu, sigma):
        # Generate normal distribution using learned mu and sigma
        z = torch.randn(bsize, self.out_dim)
        return z * torch.exp(0.5 * sigma) + mu

class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_dim, inter_dim) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = VEncoder(in_dim, [32, 128, 256, 256, 64], inter_dim)
        self.decoder = VDecoder(inter_dim, [64, 256, 256, 128, 32], in_dim)

    def forward(self, x):
        bsize = x.shape[0]
        mu_z, sigma_z = self.encoder(x)
        z = self.encoder.reparametrize(bsize, mu_z, sigma_z)
        mu_x, sigma_x = self.decoder(z)
        x_rec = self.decoder.reparametrize(bsize, mu_x, sigma_x)
        return mu_z, sigma_z, z, x_rec

    def encode(self, x):
        # Method for test-only execution
        bsize = x.shape[0]
        mu_z, sigma_z = self.encoder(x)
        z_encoded = self.encoder.reparametrize(bsize, mu_z, sigma_z)
        return z_encoded.detach().numpy()

    def decode(self, z: torch.Tensor):
        # Method for test-only execution
        bsize = z.shape[0]
        mu_x, sigma_x = self.decoder(z)
        x_rec = self.decoder.reparametrize(bsize, mu_x, sigma_x)
        return x_rec.detach().numpy()
