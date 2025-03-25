import numpy as np
import torch
def sample_process(data):
    observations, actions, rewards, next_observations, dones = data

    observations = np.array(observations)
    actions = np.array(actions)

    rewards = np.array(rewards)
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    rewards = (rewards - reward_mean) / (reward_std + 1e-9)

    next_observations = np.array(next_observations)
    dones = np.array(dones)

    return (
        torch.FloatTensor(observations),
        torch.LongTensor(actions),
        torch.FloatTensor(rewards),
        torch.FloatTensor(next_observations),
        torch.FloatTensor(dones),
    )