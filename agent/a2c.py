import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class A2C:
    def __init__(self, model, **kwargs):
        self.model = model
        self.train_step = 0
        self.gamma = kwargs.get('gamma', 0.99)
        self.use_importance = kwargs.get('use_importance', True)

    def get_action(self, observation, evaluate=False):
        with torch.no_grad():
            action_prob, state_values = self.model(observation)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)
                return action.cpu().numpy(), None
            else:
                action = action_dist.sample()
                log_probs = action_dist.log_prob(action)
                return action.cpu().numpy(), {"log_probs":log_probs.cpu().numpy(),}
        # return action.cpu().numpy(), None


    def process_trajectory(self, trajectory):
        obs = trajectory['observations'][:-1]  # 去掉最后一个终止状态
        next_obs = trajectory['observations'][1:]  # 对应 obs 的下一个状态
        actions = trajectory['actions'][:-1]
        log_probs = trajectory['log_probs'][:-1]
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        transitions = list(zip(obs, actions, log_probs, rewards, next_obs, dones))
        return transitions


    def train(self, data, weights=None):
        obs, actions, old_log_probs, rewards, nxt_obs, dones = data

        action_probs, state_values = self.model(obs)
        next_state_values, next_state_values= self.model(nxt_obs)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        target_values = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = target_values - state_values
        td_errors = advantages.detach().abs().cpu().numpy()
        if self.use_importance:
            importance = torch.exp(log_probs - old_log_probs).detach()
            actor_loss = (-log_probs * advantages.detach() * importance).mean()
        else:
            actor_loss = (-log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(state_values, target_values.detach())
        if weights is not None:
            critic_loss = weights * critic_loss
        loss = actor_loss + critic_loss
        loss = loss.mean().unsqueeze(-1)

        self.model.backward(loss)
        self.model.step()
        self.train_step += 1

        return (self.train_step,
                loss.detach().cpu().numpy(),
                td_errors)



