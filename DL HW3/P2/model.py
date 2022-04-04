import torch.nn as nn
import torch

class Encoder(nn.Module):
    '''
    Simple 2-layer MLP encoder network
    '''
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, out_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    '''
    Simple 2-layer MLP decoder network
    '''
    def __init__(self, in_dim, latent_dim, out_dim) -> None:
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, out_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SWAutoEncoder(nn.Module):
    '''
    Encoder+Decoder
    '''
    def __init__(self, in_dim, inter_dim) -> None:
        super(SWAutoEncoder, self).__init__()
        self.encoder = Encoder(in_dim, 32, inter_dim)
        self.decoder = Decoder(inter_dim, 32, in_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

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