import numpy as np
import torch
def sample_to_tensor(data):
    observations, actions, rewards, next_observations, dones = data

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    return (
        torch.FloatTensor(observations),
        torch.LongTensor(actions),
        torch.FloatTensor(rewards),
        torch.FloatTensor(next_observations),
        torch.FloatTensor(dones),
    )