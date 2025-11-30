import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import Network


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function=torch.tanh,
                 last_activation=None, trainable_std=False):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.sigma = nn.Linear(hidden_dim, output_dim)
        self.activate_func = activation_function

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)
        orthogonal_init(self.sigma, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.mean_layer(s)
        std = F.softplus(self.sigma(s)) + 0.001
        return mean, std


class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation=None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation)

    def forward(self, *x):
        x = torch.cat(x, -1)
        return self._forward(x)
