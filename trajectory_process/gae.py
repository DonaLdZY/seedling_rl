import numpy as np
from functools import partial

def get_trajectory_process(gamma = 0.99, lamda = 0.95):
    return partial(gae_process, gamma = gamma, lamda = lamda)

def gae_process(trajectory, gamma = 0.99, lamda = 0.95):
    obs = trajectory['observations']
    episode_len = len(obs) - 1
    next_obs = obs[1:]
    obs = obs[:-1]
    actions = trajectory['actions']
    actions = actions[:-1]
    rewards = trajectory['rewards']
    dones = np.zeros(episode_len, dtype=np.float32)
    dones[-1] = 1.0
    log_probs = trajectory['log_probs']
    values = trajectory['values']
    returns = np.zeros(episode_len, dtype=np.float32)
    advantages = np.zeros(episode_len, dtype=np.float32)
    last_gae = 0
    # 使用最后一个状态的 value 作为 bootstrap 值
    next_value = values[-1]
    for t in reversed(range(episode_len)):
        # delta = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lamda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]

    transitions = list(zip(obs, actions, log_probs, returns, advantages))
    return transitions