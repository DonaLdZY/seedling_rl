import torch
import torch.optim as optim

from buffer.fifo_buffer import FIFOBuffer as Buffer

from envs.lunarlander_v3.info import random_move, n_observation, n_action
# from envs.cartpole_v1.info import random_move, n_observation, n_action

from model.sac import SAC
from network.actor_net import ActorNet as ActorNetwork
from network.qnet import QNet as CriticNetwork
from trajectory_process.transitions import get_trajectory_process
from utils.sample_to_tensor import sample_to_tensor

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    critic_1_network = CriticNetwork(n_observation,  n_action, 128).to(device)
    critic_2_network = CriticNetwork(n_observation, n_action, 128).to(device)
    actor_network = ActorNetwork(n_observation, n_action, 128).to(device)

    agent_args = {
        'device': device,
        'critic_1_optimizer': optim.Adam(
            critic_1_network.parameters(),
            lr=0.001,
        ),
        'critic_2_optimizer': optim.Adam(
            critic_2_network.parameters(),
            lr=0.001,
        ),
        'actor_optimizer': optim.Adam(
            actor_network.parameters(),
            lr=0.00003,
        ),
        'random_move': random_move,
        'save_name': "test",
    }
    agent = SAC(actor_network, critic_1_network, critic_2_network, agent_args)

    buffer = Buffer(capacity=32768, startup=512)
    from learner.learner import Learner, run_learner

    learner = Learner(agent, buffer, get_trajectory_process(), sample_to_tensor, 256)
    run_learner(learner)
