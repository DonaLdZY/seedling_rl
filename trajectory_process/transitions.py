import numpy as np
def get_trajectory_process():
    return transitions_process

def transitions_process(trajectory):
    obs = trajectory['observations']
    episode_len = len(obs) - 1
    next_obs = obs[1:]
    obs = obs[:-1]
    actions = trajectory['actions']
    actions = actions[:-1]
    rewards = trajectory['rewards']
    dones = np.zeros(episode_len, dtype=np.float32)
    dones[-1] = 1.0
    transitions = list(zip(obs, actions, rewards, next_obs, dones))
    return transitions