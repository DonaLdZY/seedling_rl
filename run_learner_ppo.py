import torch
import torch.optim as optim
# from buffer.fifo_buffer import FIFOBuffer as Buffer
from buffer.prioritized_replay_buffer import PrioritizedReplayBuffer as Buffer
from envs.cartpole_v1.info import n_observation, n_action
from agent.ppo import PPO
from network.actor_net import ActorNet
from network.critic_net import CriticNet
# from trajectory_process.gae import get_trajectory_process
from trajectory_process.transitions import get_trajectory_process
from agent.a2c import A2C
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    critic = CriticNet(n_observation, 3, 128)
    actor = ActorNet(n_observation, n_action, 3, 128)
    agent_args = {
        'device': "cpu",
        'actor_optimizer': "adam",
        'actor_optimizer_args' :{
            'lr':0.001
        },
        'critic_optimizer': "adam",
        'critic_optimizer_args': {
            'lr': 0.001
        },
        'save_name': "test",
    }
    # agent = PPO(actor, critic, **agent_args)
    agent = A2C(actor, critic, **agent_args)
    buffer = Buffer(capacity=32768, startup=128)
    from learner.learner import Learner

    learner = Learner(agent,
                      buffer,
                      get_trajectory_process(),
                      pred_batching=True,
                      pred_batch_size=64,
                      use_redis=True)
    learner.run("50051", 64)
