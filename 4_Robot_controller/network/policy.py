import os
import torch
from torch import nn
from torch.distributions import Normal

from .base import BaseNetwork, create_linear_network, weights_init_xavier
from .latent import Encoder, TaskNetwork


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, input_dim, low_action_dim, hidden_units=[256, 256, 256],
                 initializer=weights_init_xavier):
        super(GaussianPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        self.net = create_linear_network(
            input_dim, low_action_dim*2, hidden_units=hidden_units,
            hidden_activation=nn.ReLU(), initializer=initializer)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x,  dim=-1)

        means, log_stds = torch.chunk(self.net(x), 2, dim=-1)
        log_stds = torch.clamp(
            log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return means, log_stds

    def sample(self, x):
        # Calculate Gaussian distribusion of (means, stds).
        # print(f'policy_shape : {x.shape}')
        means, log_stds = self.forward(x)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # Sample actions.
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # Calculate expectations of entropies.
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)

    def sample_repeat(self, x, num_repeat=10):
        # Calculate Gaussian distribusion of (means, stds).
        # print(f'policy_shape : {x.shape}')

        x_temp = x.unsqueeze(1).repeat(1, num_repeat, 1).view(x.shape[0] * num_repeat, x.shape[1])
        
        means, log_stds = self.forward(x_temp)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # Sample actions.
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # Calculate expectations of entropies.
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies.view(x.shape[0], num_repeat, 1), torch.tanh(means)

class BCPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, input_dim, low_action_dim, hidden_units=[256, 256, 256],
                 initializer=weights_init_xavier):
        super(BCPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        self.net = create_linear_network(
            input_dim, low_action_dim*2, hidden_units=hidden_units,
            hidden_activation=nn.ReLU(), initializer=initializer)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x,  dim=-1)

        means, log_stds = torch.chunk(self.net(x), 2, dim=-1)
        log_stds = torch.clamp(
            log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return means, log_stds

    def sample(self, x):
        # Calculate Gaussian distribusion of (means, stds).
        # print(f'policy_shape : {x.shape}')
        means, log_stds = self.forward(x)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # Sample actions.
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # Calculate expectations of entropies.
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        log_prob = log_probs.sum(dim=1, keepdim=True)

        return actions, log_prob, torch.tanh(means),

    def sample_repeat(self, x, num_repeat=10):
        # Calculate Gaussian distribusion of (means, stds).
        # print(f'policy_shape : {x.shape}')

        x_temp = x.unsqueeze(1).repeat(1, num_repeat, 1).view(x.shape[0] * num_repeat, x.shape[1])
        
        means, log_stds = self.forward(x_temp)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # Sample actions.
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # Calculate expectations of entropies.
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        log_prob = log_probs.sum(dim=1, keepdim=True)

        return actions, log_prob.view(x.shape[0], num_repeat, 1), torch.tanh(means)
    
class DeterministicPolicy(BaseNetwork):
    eps = 1e-6

    def __init__(self, input_dim, low_action_dim, hidden_units=[256, 256, 256],
                 initializer=weights_init_xavier):
        super(DeterministicPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        self.net = create_linear_network(
            input_dim, low_action_dim, hidden_units=hidden_units,
            hidden_activation=nn.ReLU(), initializer=initializer)
        
        self.policy_noise = 0.2
        self.noise_clip = 0.5

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x,  dim=-1)

        means = self.net(x)

        return means

    def sample(self, x):
        # Calculate Gaussian distribusion of (means, stds).
        # print(f'policy_shape : {x.shape}')
        means = torch.tanh(self.forward(x))
        noise = (torch.randn_like(means) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        noise_action = means+noise
        noise_action = noise_action.clamp(-1.0+self.eps, 1.0-self.eps)

        return means, 1, noise_action

    def sample_repeat(self, x, num_repeat=10):
        # Calculate Gaussian distribusion of (means, stds).
        x_temp = x.unsqueeze(1).repeat(1, num_repeat, 1).view(x.shape[0] * num_repeat, x.shape[1])
        
        means = torch.tanh(self.forward(x_temp))
        noise = (torch.randn_like(means) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        noise_action = means+noise
        noise_action = noise_action.clamp(-1.0+self.eps, 1.0-self.eps)

        return means, 1, noise_action


class EvalPolicy(BaseNetwork):

    def __init__(self, observation_shape, action_shape, num_sequences=8,
                 feature_dim=256, hidden_units=[256, 256], leaky_slope=0.2):
        super(EvalPolicy, self).__init__()
        self.encoder = Encoder(
            observation_shape[0], feature_dim, leaky_slope=leaky_slope)
        self.policy = GaussianPolicy(
            num_sequences * feature_dim
            + (num_sequences-1) * action_shape[0],
            action_shape[0], hidden_units)

    def forward(self, states, actions):
        num_batches = states.size(0)

        features = self.latent.encoder(states).view(num_batches, -1)
        actions = torch.FloatTensor(actions).view(num_batches, -1)
        feature_actions = torch.cat([features, actions], dim=-1)

        means, log_stds = torch.chunk(self.net(feature_actions), 2, dim=-1)
        log_stds = torch.clamp(
            log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return means, log_stds

    def sample(self, states, actions):
        # Calculate Gaussian distribusion of (means, stds).
        means, log_stds = self.forward(states, actions)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # Sample actions.
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # Calculate expectations of entropies.
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)

    def load_weights(self, model_dir):
        self.encoder.load(os.path.join(model_dir, 'encoder.pth'))
        self.policy.load(os.path.join(model_dir, 'policy.pth'))

