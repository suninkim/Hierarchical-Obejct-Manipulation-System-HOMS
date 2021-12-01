from collections import deque
import numpy as np
import torch
from torch.distributions.kl import kl_divergence


def create_feature_actions(features_seq, tasks_seq, actions_seq):
    N = features_seq.size(0)

    # sequence of features
    f = features_seq[:, :-1].view(N, -1)
    n_f = features_seq[:, 1:].view(N, -1)

    # sequence of tasks
    t = tasks_seq[:, :-1].view(N, -1)
    n_t = tasks_seq[:, 1:].view(N, -1)

    # sequence of actions
    a = actions_seq[:, :-1].view(N, -1)
    n_a = actions_seq[:, 1:].view(N, -1)

    # feature_actions
    fta = torch.cat([f, t, a], dim=-1)
    n_fta = torch.cat([n_f, n_t, n_a], dim=-1)

    return fta, n_fta


def calc_kl_divergence(p_list, q_list):
    assert len(p_list) == len(q_list)

    kld = 0.0
    for i in range(len(p_list)):
        # (N, L) shaped array of kl divergences.
        _kld = kl_divergence(p_list[i], q_list[i])
        # Average along batches, sum along sequences and elements.
        kld += _kld.mean(dim=0).sum()

    return kld


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False

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
