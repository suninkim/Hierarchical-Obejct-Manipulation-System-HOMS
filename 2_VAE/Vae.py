import argparse
import os
import numpy as np
import time
import datetime
import sys
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

from vae_model import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="epoch to start training from")
parser.add_argument("--dataset_name", type=str, default="Vae", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=3, help="interval between model checkpoints")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()
print(opt)

epochs = opt.epochs
batch_size = opt.batch_size    
beta = 1e-8

train_ds = OfflineDataset(img_dim=216)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=opt.n_cpu)
valid_dataloader = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=1)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
print(device)

model = LatentNetwork()
model.to(device)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

step = 0
for epoch in range(epochs):
    model.train()

    for i, image in enumerate(train_dataloader):
        optimizer.zero_grad()

        image = image.to(device)
        image_recon, recon_loss, kl_loss = model(image)
        loss = recon_loss + beta*kl_loss

        sys.stdout.write("\r---- [Epoch %d/%d, Step %d/%d] loss: %.6f----" % (
            epoch, epochs, i+1, len(train_dataloader), loss.item())
        )

        loss.backward()
        optimizer.step()

        step += 1

        if step % opt.sample_interval == 0:

            with torch.no_grad():
                images = next(iter(valid_dataloader)).to(device)

                gen_img = model.test(images)
                cat_img = torch.cat((images,gen_img),dim=2)

                save_image(cat_img.data, f"images/{opt.dataset_name}/img_{step}.png", nrow=3, normalize=True)


    torch.save(model.state_dict(), f"saved_models/{opt.dataset_name}/ckpt_{epoch}.pth")


