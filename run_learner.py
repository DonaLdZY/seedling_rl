import torch
import torch.optim as optim
from agent.DQN import DQN
from actor.CartPole_v1.network.QNet import Network
from actor.CartPole_v1.env_info import random_move
from buffer.replay_buffer import ReplayBuffer

from learner.learner import run_learner
if __name__ == "__main__":
    save_name = "test_test"
    model = Network()
    agent_args = {
        'device': torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0'),
        'eval_mode': False,
        'optimizer': optim.Adam(
            model.parameters(),
            lr=0.0003,
            weight_decay=0.0005
        ),
        'random_move': random_move,
        'save_name': save_name,
    }
    agent = DQN(Network(), agent_args)
    buffer = ReplayBuffer(capacity=1024)
    run_learner(agent, buffer)