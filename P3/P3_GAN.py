import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plot
import os

from data import SwissRollDataset
from model import Generator, Discriminator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyperparameters
N = 2500            # Number of data points
EPOCH = 5000         # Number of epochs
BSIZE = 250         # Batch size
D_ITER = 5

# Dataset
dataset = SwissRollDataset(N)
dataloader = DataLoader(dataset, batch_size=BSIZE, shuffle=False)

# Model
G_model = Generator(2, [32, 256, 512, 512, 256, 32], 3)
D_model = Discriminator(3, [32, 256, 512, 512, 256, 32], 1)

# Reconstruction loss
rec_loss = nn.MSELoss()

# Optimizer
G_optim = torch.optim.Adam(G_model.parameters(), lr=0.0001)
D_optim = torch.optim.Adam(D_model.parameters(), lr=0.0001)

# Loss value records (for plotting)
g_loss = []
d_loss = []

# Training
for e in range(EPOCH):
    for i, data in enumerate(dataloader):
        z = torch.randn(BSIZE, 2)

        # generator forward
        x_gen = G_model(z)

        # discriminator iters
        for _ in range(D_ITER):
            # discriminator forward
            p_true = D_model(data)
            p_false = D_model(x_gen)
            
            # CE loss
            disc_ce_loss = -torch.mean(torch.log(p_true)) - torch.mean(torch.log(1 - p_false))

            # discriminator backward
            D_optim.zero_grad()
            disc_ce_loss.backward(retain_graph=True)
            D_optim.step()

            # discriminator logging
            d_loss.append(disc_ce_loss.item())
        
        # generator loss
        p_false = D_model(x_gen)
        gen_loss = torch.mean(torch.log(1 - p_false))

        # generator backward
        G_optim.zero_grad()
        gen_loss.backward()
        G_optim.step()
        g_loss.append(gen_loss.item())

        # generator logging
        print(f'Epoch {e} {i * BSIZE}/{len(dataset)}:', 
              f'G_loss={round(gen_loss.item(), 4)},',
              f'D_loss={round(disc_ce_loss.item(), 4)}')

# Plot loss curve
x1 = np.arange((len(d_loss)))
y1 = np.array(d_loss)
x2 = np.arange((len(g_loss))) * D_ITER
y2 = np.array(g_loss)

plot.figure()
plot.plot(x1, y1, 'b-')
plot.plot(x2, y2, 'g-')
plot.xlabel('Iter')
plot.ylabel('Loss_val')
plot.legend(['Discriminator loss', 'Generator loss'])
plot.savefig('GAN/loss.png')

# Plot encoded data
full_z = torch.randn(len(dataset), 2)
generation_final = G_model(full_z).detach().numpy()
fig = plot.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(generation_final[:, 0], generation_final[:, 1], generation_final[:, 2], marker='o')
ax.set_title('Generated data')
plot.savefig('GAN/generated.png')

# Show original data
fig = plot.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], marker='o')
ax.set_title('Original data')
plot.savefig('GAN/original.png')
