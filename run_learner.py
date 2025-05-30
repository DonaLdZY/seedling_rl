import torch
from buffer.prioritized_replay_buffer import PrioritizedReplayBuffer as Buffer
from envs.lunarlander_v3.info import n_observation, n_action
from agent.vtrace_fake import VTrace
# from agent.ppo import PPO
from agent.a2c import A2C
from model.model import Model
from network.actor_critic import ActorCritic
from network.actor_net import ActorNet

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(n_observation, n_action, 3, 128)
    model_args = {
        'device': "cuda:0" if torch.cuda.is_available() else "cpu",
        'optimizer': {
            "type": "Adam",
            'params' :{
                'lr':0.0003
            }
        },
    }
    model = Model(model, **model_args)
    agent = VTrace(model)
    buffer = Buffer(capacity=4096, startup=128, max_usage=-1)
    from learner.learner import Learner

    learner = Learner(agent,
                      buffer,
                      pred_batching=True,
                      pred_batch_size=512,
                      auto_batching=True,
                      use_redis=False,)
    learner.run("50051", 64)
