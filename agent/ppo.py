import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class PPO:
    def __init__(self, model, **kwargs):
        self.model = model
        self.train_step = 0
        self.gamma = kwargs.get('gamma', 0.99)
        self.lambda_ = kwargs.get('lambda_', 0.95)
        self.clip_param = kwargs.get('clip_param', 0.2)


    def get_action(self, observation, evaluate=False):
        with (torch.no_grad()):
            action_prob, state_values = self.model(observation)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)
                return action.cpu().numpy(), None
            else:
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                return action.cpu().numpy(), \
                    {
                        "log_probs":log_prob.cpu().numpy(),
                        "values":state_values.cpu().numpy()
                    }

    def process_trajectory(self, trajectory):
        obs = trajectory['observations'][:-1]
        next_obs = trajectory['observations'][1:]
        actions = trajectory['actions'][:-1]
        log_probs = trajectory['log_probs'][:-1]
        rewards = trajectory['rewards']
        values = trajectory['values']
        episode_len = len(obs)
        dones = np.zeros(episode_len, dtype=np.float32)
        dones[-1] = 1.0
        returns = np.zeros(episode_len, dtype=np.float32)
        advantages = np.zeros(episode_len, dtype=np.float32)
        last_gae = 0
        for t in reversed(range(episode_len)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
        transitions = list(zip(obs, actions, log_probs, returns, advantages))
        return transitions


    def train(self, data, weights=None):
        observations, actions, old_log_probs, returns, advantages = data

        action_probs, state_values = self.model(observations)
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs) # 计算概率比率
        # PPO 剪切目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = F.mse_loss(state_values, returns)
        td_errors = (returns - state_values.detach()).abs().cpu().numpy()
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


