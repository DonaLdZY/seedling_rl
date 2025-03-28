import torch
import torch.optim as optim
from buffer.fifo_buffer import FIFOBuffer as Buffer
from envs.cartpole_v1.info import random_move, n_observation, n_action
from model.dqn import DQN
from network.qnet_dueling import QNetDueling
from trajectory_process.transitions import get_trajectory_process

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = QNetDueling(n_observation, n_action, 3, 128).to(device)
    agent_args = {
        'device': device,
        'optimizer': optim.Adam(
            network.parameters(),
            lr=0.001,
        ),
        'random_move': random_move,
        'save_name': "test",
        'epsilon_max': 0.08,
        'epsilon_decay': 0.9995,
        'epsilon_min': 0.000,
    }
    agent = DQN(network, agent_args)

    buffer = Buffer(capacity=32768, startup=1024)
    from learner.learner import Learner

    learner = Learner(agent, buffer, get_trajectory_process(), pred_batching=True, pred_batch_size=64)
    learner.run("50051", 64)
