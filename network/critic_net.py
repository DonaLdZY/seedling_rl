from torch import nn

class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_num=3, hidden_size = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(hidden_num - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))  # 创建新实例
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

