import numpy as np
import torch
import threading
from torch.distributions import Categorical
from utils.utils import normalize

class PolicyGradient:
    def __init__(self, model, **kwargs):
        self.model = model
        self.train_step = 0
        self.lock = threading.Lock()
        self.gamma = kwargs.get('gamma', 0.99)
        self.use_importance = kwargs.get('use_importance', True)

    def get_action(self, observation, evaluate=False):
        with torch.no_grad():
            action_prob = self.model(observation)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)
                return action.cpu().numpy(), None
            else:
                action = action_dist.sample()
                log_probs = action_dist.log_prob(action)
                return action.cpu().numpy(), {"log_probs": log_probs.cpu().numpy(), }


    def process_trajectory(self, trajectory):
        obs = trajectory['observations'][:-1]  # 去掉最后一个终止状态
        actions = trajectory['actions'][:-1]
        log_probs = trajectory['log_probs'][:-1]
        rewards = trajectory['rewards']
        episode_length = len(obs)
        advantage = np.zeros_like(rewards, dtype=np.float32)
        G = 0.0
        for t in reversed(range(episode_length)):
            G = rewards[t] + self.gamma * G
            advantage[t] = G
        transitions = list(zip(obs, actions, log_probs, advantage))
        return transitions


    def train(self, data, weights=None):
        obs, actions, old_log_probs, advantages = data
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        action_probs = self.model(obs)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        if self.use_importance:
            importance = torch.exp(log_probs - old_log_probs).detach()
            loss = (-log_probs * advantages.detach() * importance).mean()
        else:
            loss = (-log_probs * advantages.detach()).mean()

        loss = loss.mean().unsqueeze(-1)

        self.model.backward(loss)
        self.model.step()
        self.train_step += 1

        return (self.train_step,
                loss.detach().cpu().numpy(),
                None)





