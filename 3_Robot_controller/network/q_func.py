import torch
from torch import nn
from torch.distributions.transforms import Transform

from .base import BaseNetwork, create_linear_network, weights_init_xavier

class QNetwork(BaseNetwork):

    def __init__(self, latent_dim, action_dim, hidden_units=[256, 256, 256],
                 initializer=weights_init_xavier):
        super(QNetwork, self).__init__()


        self.Q = create_linear_network(
            latent_dim + action_dim, 1, hidden_units=hidden_units,
            hidden_activation=nn.ReLU(), initializer=initializer)

    def forward(self, x):
        q = self.Q(x)
        return q


class TwinnedQNetwork(BaseNetwork):

    def __init__(self, latent_dim, low_action_dim, hidden_units=[256, 256],
                 initializer=weights_init_xavier):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(
            latent_dim, low_action_dim, hidden_units, initializer)
        self.Q2 = QNetwork(
            latent_dim, low_action_dim, hidden_units, initializer)        

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2

    def forward_repeat(self, states, actions, num_repeat=10):

        states_temp = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])

        x = torch.cat([states_temp, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)

        q1 = q1.view(states.shape[0], num_repeat, 1)
        q2 = q2.view(states.shape[0], num_repeat, 1)

        return q1, q2

