import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=240, help="size of image height")
parser.add_argument("--img_width", type=int, default=240, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=3, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss().cuda()
criterion_pixelwise = torch.nn.L1Loss().cuda()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet().cuda()
discriminator_can = Discriminator(dex='can').cuda()
discriminator_seg = Discriminator(dex='seg').cuda()
discriminator_dep = Discriminator(dex='dep').cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_can.load_state_dict(torch.load("saved_models/%s/discriminator_can_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_seg.load_state_dict(torch.load("saved_models/%s/discriminator_seg_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator_dep.load_state_dict(torch.load("saved_models/%s/discriminator_dep_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator_can.apply(weights_init_normal)
    discriminator_seg.apply(weights_init_normal)
    discriminator_dep.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_can = torch.optim.Adam(discriminator_can.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_dep = torch.optim.Adam(discriminator_dep.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_seg = torch.optim.Adam(discriminator_seg.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset(),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = imgs["A"].cuda()
    real_B_can = imgs["B"].cuda()
    real_B_seg = imgs["C"].cuda()
    real_B_dep = imgs["D"].cuda()
    real_B_seg = real_B_seg[:,:3,:,:] + 0.3*real_B_seg[:,3:6,:,:] + 0.6*real_B_seg[:,6:9,:,:]
    real_B_dep = torch.cat((real_B_dep,real_B_dep,real_B_dep),axis=1)


    fake_B_can, fake_B_seg, fake_B_dep = generator(real_A)
    fake_B_seg = fake_B_seg[:,:3,:,:] + 0.3*fake_B_seg[:,3:6,:,:] + 0.6*fake_B_seg[:,6:9,:,:]
    fake_B_dep = torch.cat((fake_B_dep,fake_B_dep,fake_B_dep),axis=1)

    img_sample = torch.cat((real_A.data, real_B_can.data, real_B_seg.data, real_B_dep.data, fake_B_can.data, fake_B_seg.data, fake_B_dep.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = batch["A"].cuda()
        real_B_can = batch["B"].cuda()
        real_B_seg = batch["C"].cuda()
        real_B_dep = batch["D"].cuda()

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B_can, fake_B_seg, fake_B_dep = generator(real_A)
        pred_fake_can = discriminator_can(fake_B_can, real_A)
        loss_GAN_can = criterion_GAN(pred_fake_can, valid)

        pred_fake_seg = discriminator_seg(fake_B_seg, real_A)
        loss_GAN_seg = criterion_GAN(pred_fake_seg, valid)

        pred_fake_dep = discriminator_dep(fake_B_dep, real_A)
        loss_GAN_dep = criterion_GAN(pred_fake_dep, valid)

        # Pixel-wise loss
        loss_pixel_can = criterion_pixelwise(fake_B_can, real_B_can)
        loss_pixel_seg = criterion_pixelwise(fake_B_seg, real_B_seg)
        loss_pixel_dep = criterion_pixelwise(fake_B_dep, real_B_dep)


        # Total loss
        loss_G = 0.33*(loss_GAN_can + loss_GAN_seg + loss_GAN_dep + lambda_pixel*(loss_pixel_can + 5*loss_pixel_seg + loss_pixel_dep))
        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D_can.zero_grad()
        optimizer_D_seg.zero_grad()
        optimizer_D_dep.zero_grad()

        # Real loss
        pred_real_can = discriminator_can(real_B_can, real_A)
        loss_real_can = criterion_GAN(pred_real_can, valid)

        # Fake loss
        pred_fake_can = discriminator_can(fake_B_can.detach(), real_A)
        loss_fake_can = criterion_GAN(pred_fake_can, fake)

        # Real loss
        pred_real_seg = discriminator_seg(real_B_seg, real_A)
        loss_real_seg = criterion_GAN(pred_real_seg, valid)

        # Fake loss
        pred_fake_seg = discriminator_seg(fake_B_seg.detach(), real_A)
        loss_fake_seg = criterion_GAN(pred_fake_seg, fake)

        # Real loss
        pred_real_dep = discriminator_dep(real_B_dep, real_A)
        loss_real_dep = criterion_GAN(pred_real_dep, valid)

        # Fake loss
        pred_fake_dep = discriminator_dep(fake_B_dep.detach(), real_A)
        loss_fake_dep = criterion_GAN(pred_fake_dep, fake)


        # Total loss
        loss_D = 0.5 * 0.33* (loss_real_can + loss_fake_can + loss_real_seg + loss_fake_seg + loss_real_dep + loss_fake_dep) 
        loss_D.backward()
        optimizer_D_can.step()
        optimizer_D_seg.step()
        optimizer_D_dep.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                (loss_pixel_can+loss_pixel_seg+loss_pixel_dep).item(),
                (loss_GAN_can+loss_GAN_seg+loss_GAN_dep).item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator_can.state_dict(), "saved_models/%s/discriminator_can_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator_seg.state_dict(), "saved_models/%s/discriminator_seg_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator_dep.state_dict(), "saved_models/%s/discriminator_dep_%d.pth" % (opt.dataset_name, epoch))
