
from itertools import pairwise
import torch
import torch.nn as nn

dims = [100, 10]

class Net(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Net, self).__init__()

        layers = [state_size] + dims + [action_size]

        weight_layers = [nn.Linear(in_feat, out_feat) for in_feat, out_feat in pairwise(layers)]
        activation_layers = [nn.ReLU() for _ in range(len(weight_layers))]
        net_layers = 2 * [None] * len(weight_layers)
        net_layers[::2] = weight_layers
        net_layers[1::2] = activation_layers

        self.seed = torch.manual_seed(seed)
        self.fc = nn.Sequential(
            *net_layers
        )

    def forward(self, x):
        return self.fc(x)
