from torch import nn
from network.actor_net import ActorNet
from network.q_net import QNet


class Actor2QNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_num = 2, hidden_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actor = ActorNet(input_dim, output_dim, hidden_num, hidden_size)
        self.q1 = QNet(input_dim, output_dim, hidden_num, hidden_size)
        self.q2 = QNet(input_dim, output_dim, hidden_num, hidden_size)

    def forward(self, x):
        actions = self.actor(x)
        values1 = self.q1(x)
        values2 = self.q2(x)
        return actions, values1, values2