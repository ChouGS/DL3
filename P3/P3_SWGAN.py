import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plot
import os

from data import SwissRollDataset
from loss import SlicedWSLoss3D
from model import Generator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyperparameters
N = 2500            # Number of data points
EPOCH = 5000         # Number of epochs
BSIZE = 250         # Batch size
L = 512
D_ITER = 5

# Dataset
dataset = SwissRollDataset(N)
dataloader = DataLoader(dataset, batch_size=BSIZE, shuffle=False)

# Model
G_model = Generator(2, [32, 256, 512, 512, 256, 32], 3)

# Reconstruction loss
rec_loss = nn.MSELoss()
pp_loss = SlicedWSLoss3D(L)

# Optimizer
G_optim = torch.optim.Adam(G_model.parameters(), lr=0.0001)

# Loss value records (for plotting)
g_loss = []

# Training
for e in range(EPOCH):
    for i, data in enumerate(dataloader):
        z = torch.randn(BSIZE, 2)

        # forward
        x_gen = G_model(z)
        
        # Sliced Wasserstein loss
        gen_loss = pp_loss(x_gen, data)

        # backward
        G_optim.zero_grad()
        gen_loss.backward()
        G_optim.step()
        g_loss.append(gen_loss.item())

        # logging
        print(f'Epoch {e} {i * BSIZE}/{len(dataset)}:', 
              f'G_loss={round(gen_loss.item(), 4)}')

# Plot loss curve
x = np.arange((len(g_loss))) * D_ITER
y = np.array(g_loss)

plot.figure()
plot.plot(x, y, 'b-')
plot.xlabel('Iter')
plot.ylabel('Loss_val')
plot.legend(['Generator loss'])
plot.savefig('SWGAN/loss.png')

# Plot encoded data
full_z = torch.randn(len(dataset), 2)
generation_final = G_model(full_z).detach().numpy()
fig = plot.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(generation_final[:, 0], generation_final[:, 1], generation_final[:, 2], marker='o')
ax.set_title('Generated data')
plot.savefig('SWGAN/generated.png')

# Show original data
fig = plot.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], marker='o')
ax.set_title('Original data')
plot.savefig('SWGAN/original.png')
