import numpy as np
from functools import partial

def get_trajectory_process(gamma = 0.98):
    return partial(decaying_reward_process, gamma = gamma)

def decaying_reward_process(trajectory, gamma=0.98):
    obs = trajectory['observations']
    episode_len = len(obs) - 1
    next_obs = obs[1:]
    obs = obs[:-1]
    actions = trajectory['actions']
    actions = actions[:-1]
    dones = np.zeros(episode_len, dtype=np.float32)
    dones[-1] = 1.0
    rewards = trajectory['rewards']
    returns = np.zeros_like(rewards, dtype=np.float32)
    G = 0.0
    for t in reversed(range(episode_len)):
        G = rewards[t] + gamma * G
        returns[t] = G
    transitions = list(zip(obs, actions, returns, next_obs, dones))
    return transitions


