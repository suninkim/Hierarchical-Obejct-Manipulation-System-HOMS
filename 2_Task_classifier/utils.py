from collections import deque
import numpy as np
import torch

def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False
        
def center_crop(cat_img, device, out=216):
    
    n, c, h, w = cat_img.shape
    crop_max = h - out + 1
    w1 = np.zeros((n,)) + crop_max//2
    h1 = np.zeros((n,)) + crop_max//2
    w1 = w1.astype(np.uint8)
    h1 = h1.astype(np.uint8)
    cropped_cat = torch.zeros((n, c, out, out), dtype=cat_img.dtype, device=device)

    for i, (img, w11, h11) in enumerate(zip(cat_img, w1, h1)):
        cropped_cat[i] = img[:, h11:h11 + out, w11:w11 + out]

    return cropped_cat
        
def rand_crop(cat_img, device, out=216):
    
    n, c, h, w = cat_img.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped_cat = torch.zeros((n, c, out, out), dtype=cat_img.dtype, device=device)
    
    for i, (img, w11, h11) in enumerate(zip(cat_img, w1, h1)):
        cropped_cat[i] = img[:, h11:h11 + out, w11:w11 + out]

    return cropped_cat

def get_aug_image(images, device):
    
    n, c, h, w = images.shape    
    images_uni = images + torch.empty((n, c, h, w), device=device).uniform_(-3e-2, 3e-2)
    images_gau = images + torch.normal(mean=torch.zeros((n, c, h, w), device=device), std=3e-2)
    images = torch.cat((images, images_uni, images, images_gau),dim=0)
    images = rand_crop(images, device)

    return images


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
