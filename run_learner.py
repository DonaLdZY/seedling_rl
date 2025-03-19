import torch
import torch.optim as optim
import asyncio
from model.DQN import DQN
from network.qnet_naive import Network
from envs.cartpole_v1.info import random_move, n_observation, n_action
from buffer.fifo_buffer import FIFOBuffer


if __name__ == "__main__":
    save_name = "test_test"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Network(n_observation, n_action).to(device)
    agent_args = {
        'device': device,
        'eval_mode': False,
        'optimizer': optim.Adam(
            model.parameters(),
            lr=0.0003,
            weight_decay=0.0005
        ),
        'random_move': random_move,
        'save_name': save_name,
    }
    agent = DQN(model, agent_args)
    buffer = FIFOBuffer(capacity=2048, startup=128)


    from learner.learner import run_learner
    run_learner(agent, buffer)


    # import asyncio
    # from learner.learner_async import run_learner
    # asyncio.run(run_learner(model, buffer))