from torch import nn
from network.actor_net import ActorNet
from network.critic_net import CriticNet
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_num = 2, hidden_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.actor = ActorNet(input_dim, output_dim, hidden_num, hidden_size)
        self.critic = CriticNet(input_dim, hidden_num, hidden_size)

    def forward(self, x):
        values = self.critic(x)
        actions = self.actor(x)
        return actions, values