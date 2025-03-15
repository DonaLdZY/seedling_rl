import torch
from torch import nn
import numpy as np

from ..env_info import n_observation, n_action

class Network(nn.Module):
    def __init__(self, hidden_size = 64):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_observation, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_action)
        )
    def forward(self, x):
        x = torch.Tensor(np.array(x))
        return self.net(x)

