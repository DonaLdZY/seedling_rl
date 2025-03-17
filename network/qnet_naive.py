from torch import nn

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_size = 64
        self.net=nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
    def forward(self, x):
        return self.net(x)

