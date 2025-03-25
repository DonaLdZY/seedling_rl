import torch
import torch.optim as optim
import asyncio

from buffer.fifo_buffer import FIFOBuffer as Buffer

from envs.lunarlander_v3.info import random_move, n_observation, n_action

from model.dqn import DQN
from network.qnet_dueling import Network
from trajectory_process.transitions import get_trajectory_process
from sample_process.default import sample_process

# from model.policy_gradient import PolicyGradient
# from network.actor_net import Network
# from trajectory_process.decaying_reward import get_trajectory_process
# from sample_process.batch_reward_normalize import sample_process

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = Network(n_observation, n_action, 256).to(device)

    agent_args = {
        'device': device,
        'optimizer': optim.Adam(
            network.parameters(),
            lr=0.001,
        ),
        'random_move': random_move,
        'save_name': "test",
        'epsilon_max': 0.8,
        'epsilon_decay': 0.9995,
        'epsilon_min': 0.000,
    }
    agent = DQN(network, agent_args)
    # agent = PolicyGradient(network, agent_args)

    buffer = Buffer(capacity=32768, startup=1024)
    from learner.learner import Learner, run_learner

    learner = Learner(agent, buffer, get_trajectory_process(), sample_process, 256)
    run_learner(learner)


    # import asyncio
    # from learner.learner_async import run_learner
    # asyncio.run(run_learner(model, buffer))