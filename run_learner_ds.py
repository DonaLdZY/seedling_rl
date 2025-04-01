import torch
from buffer.prioritized_replay_buffer import PrioritizedReplayBuffer as Buffer
from envs.cartpole_v1.info import n_observation, n_action
from agent.ppo import PPO
from model.model import Model
from network.actor_critic import ActorCritic
import os

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(n_observation, n_action, 4096, 128)
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
    os.environ["USE_LIBUV"] = "0"
    # import deepspeed
    # model, optimizer, _, _ = deepspeed.initialize(
    #     model=model,
    #     config="ds_zero2.json"  # DeepSpeed配置文件
    # )
    agent = PPO(model)
    buffer = Buffer(capacity=8, startup=1, max_usage=1)
    from learner.learner import Learner
    learner = Learner(agent,
                      buffer,
                      pred_batching=True,
                      pred_batch_size=64,
                      train_batch_size=1,
                      use_redis=False,)
    torch.cuda.empty_cache()
    learner.run("50051", 64)
