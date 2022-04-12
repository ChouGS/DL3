import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plot
import os

from loss import VAE_KL_loss
from data import SwissRollDataset
from model import VariationalAutoEncoder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyperparameters
N = 2500            # Number of data points
EPOCH = 1000         # Number of epochs
BSIZE = 100         # Batch size
EPS = 0.05

# Dataset
dataset = SwissRollDataset(N)
dataloader = DataLoader(dataset, batch_size=BSIZE, shuffle=False)

# Model
vae = VariationalAutoEncoder(3, 2)

# Reconstruction loss
rec_loss = nn.MSELoss()

# Distribution loss (KL-Divergence)
pp_loss = VAE_KL_loss()

# Optimizer
optim = torch.optim.Adam(vae.parameters(), lr=0.005)

# Loss value records (for plotting)
rec = []
pp = []

# Training
for e in range(EPOCH):
    for i, data in enumerate(dataloader):
        # forward
        mu_z, gamma_z, z_encoded, x_rec = vae(data)

        rec_loss_val = rec_loss(x_rec, data)
        pp_loss_val = pp_loss(mu_z, gamma_z)
        loss_val = rec_loss_val + EPS * pp_loss_val

        # backward
        optim.zero_grad()
        loss_val.backward()
        optim.step()

        # logging
        rec.append(rec_loss_val.item())
        pp.append(pp_loss_val.item() * EPS)

        # print status
        print(f'Epoch {e} {i * BSIZE}/{len(dataset)}:', 
              f'Rec_loss={round(rec_loss_val.item(), 4)},',
              f'PP_loss={round(pp_loss_val.item() * EPS, 4)}')

# Plot loss curve
x = np.arange((len(rec)))
y1 = np.array(rec)
y2 = np.array(pp)

plot.figure()
plot.subplot(1, 2, 1)
plot.plot(x, y1, 'b-')
plot.xlabel('Iter')
plot.ylabel('Loss_val')
plot.legend(['reconstruction loss'])
plot.subplot(1, 2, 2)
plot.plot(x, y2, 'g-')
plot.xlabel('Iter')
plot.ylabel('Loss_val')
plot.legend([f'distribution loss * {EPS}'])
plot.savefig('VAE/loss.png')

# Plot encoded data
full_data = dataset[:]
z_final = vae.encode(full_data)
fig = plot.figure()
plot.scatter(z_final[:, 0], z_final[:, 1])
plot.title('Encoded distribution')
plot.savefig('VAE/distribution.png')

# Testing - use decoder to generte Swiss roll
z_test = torch.randn(N, 2)
generated_data = vae.decode(z_test)

# Plot generated data and original data
fig = plot.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], marker='o')
ax.set_title('Generated data')
plot.savefig('VAE/generated.png')

fig = plot.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], marker='o')
ax.set_title('Original data')
plot.savefig('VAE/original.png')
