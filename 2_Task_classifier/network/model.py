import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .base import BaseNetwork

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# class BaseNetwork(nn.Module):
#     def save(self, path):
#         torch.save(self.state_dict(), path)

#     def load(self, path):
#         self.load_state_dict(torch.load(path))


class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)

class DQNPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, state_shape, shared=True,
                 dueling_net=True):
        super().__init__()

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                # nn.Linear(7 * 7 * 64, 512),
                nn.Linear(state_shape, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(state_shape, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(state_shape, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)
        
    def act(self, states):
        
        if not self.shared:
            states = self.conv(states)
        if not self.dueling_net:
            action_logits = self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            action_logits = v + a - a.mean(1, keepdim=True)

        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions


class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, state_shape, shared=True,
                 dueling_net=True):
        super().__init__()

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                # nn.Linear(7 * 7 * 64, 512),
                nn.Linear(state_shape, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(state_shape, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(state_shape, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)



class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_actions, state_shape, shared=True,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, num_actions, state_shape, shared, dueling_net)
        self.Q2 = QNetwork(num_channels, num_actions, state_shape, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, state_shape, num_actions, shared=True):
        super().__init__()
        if not shared:
            self.conv = DQNBase(num_channels)

        self.head = nn.Sequential(
            nn.Linear(state_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions))

        self.shared = shared

    def act(self, states):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        if not self.shared:
            states = self.conv(states)

        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
    
class Classifier(BaseNetwork):
    
    def __init__(self, num_channels, state_shape, num_actions, shared=True):
        super().__init__()
        if not shared:
            self.conv = DQNBase(num_channels)

        self.head = nn.Sequential(
            nn.Linear(state_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions))


        self.shared = shared

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        action_probs = F.softmax(self.head(states), dim=1)

        return action_probs

    def act(self, states):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

