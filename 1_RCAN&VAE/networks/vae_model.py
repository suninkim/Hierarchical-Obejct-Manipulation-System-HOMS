import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

from .base import BaseNetwork, create_linear_network, weights_init_xavier

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Gaussian(BaseNetwork):

    def __init__(self, input_dim, output_dim, hidden_units=[256, 256],
                 std=None, leaky_slope=0.2):
        super(Gaussian, self).__init__()
        self.net = create_linear_network(
            input_dim, 2*output_dim if std is None else output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(leaky_slope),
            initializer=weights_init_xavier)

        self.std = std

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)
        # print(f"gau_shape:{x.shape}")
        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        return Normal(loc=mean, scale=std)


class ConstantGaussian(BaseNetwork):

    def __init__(self, output_dim, std=1.0):
        super(ConstantGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros((x.size(0), self.output_dim)).to(x)
        std = torch.ones((x.size(0), self.output_dim)).to(x) * self.std
        return Normal(loc=mean, scale=std)

class Decoder(BaseNetwork):

    def __init__(self, input_dim=256, output_dim=3, std=1.0, leaky_slope=0.2, bot_dim = 10):
        super(Decoder, self).__init__()
        self.std = std
        self.leaky_slope = leaky_slope

        self.convts = nn.ModuleList(
            [nn.ConvTranspose2d(input_dim, 64, 27),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1),
            nn.ConvTranspose2d(32, 32, 5, 2, 2, 1),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, output_dim, 3, 1, 1)]
            )

        self.apply(weight_init)

    def forward(self, x):

        num_batches, latent_dim = x.size()

        x = x.view(num_batches, latent_dim, 1, 1)

        for i in range(8):
            x = F.leaky_relu(self.convts[i](x),negative_slope=self.leaky_slope)

        con = torch.tanh(self.convts[8](x))

        return Normal(loc=con, scale=torch.ones_like(con) * self.std)

class Encoder(BaseNetwork):

    def __init__(self, input_dim=3, feature_dim=256, latent_dim=256, hidden_units=[256, 256], leaky_slope=0.2):
        super(Encoder, self).__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 1)]
            )

        self.leaky_slope = leaky_slope

        self.gau = Gaussian(
            27*27*64, latent_dim, hidden_units, leaky_slope=leaky_slope)

        self.apply(weight_init)

    def forward(self, x):

        num_batches, C, H, W = x.size()

        for i in range(6):
            x = F.leaky_relu(self.convs[i](x),negative_slope=self.leaky_slope)

        x = x.view(num_batches,-1)

        dist = self.gau(x)

        return dist.loc, dist

class LatentNetwork(BaseNetwork):

    def __init__(self, observation_shape=(3,256,256), feature_dim=256,
                 latent_dim=256, hidden_units=[256, 256],
                 leaky_slope=0.2):
        super(LatentNetwork, self).__init__()

        self.encoder = Encoder(observation_shape[0], feature_dim, latent_dim, hidden_units, leaky_slope=leaky_slope)
        self.decoder = Decoder(latent_dim, observation_shape[0], std=np.sqrt(0.1), leaky_slope=leaky_slope)
        
    def forward(self, x):

        feature, dist = self.encoder(x)
        dist_sample = dist.rsample()

        mu = dist.loc
        logvar = dist.scale

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        recon_img_dists = self.decoder(
            dist_sample)

        log_likelihood_loss = recon_img_dists.log_prob(
            x).mean(dim=0).sum()

        return recon_img_dists.loc, -log_likelihood_loss, KLD

    def test(self, x):

        feature, dist = self.encoder(x)
        recon_img_dists = self.decoder(feature)

        return recon_img_dists.loc

