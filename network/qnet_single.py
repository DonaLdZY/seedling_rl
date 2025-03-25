from torch import nn, cat
import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net=nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, state, action):
        action_one_hot = F.one_hot(action, self.action_dim).float()
        x = cat((state, action_one_hot), dim=-1)
        return self.net(x).squeeze(-1)

