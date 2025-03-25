from torch import nn

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super().__init__()
        self.output_dim = output_dim

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 输出单个状态价值
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)  # 输出每个动作的优势值
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values