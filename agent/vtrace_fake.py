import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical



class VTrace:
    def __init__(self, model, **kwargs):
        self.model = model
        self.train_step = 0
        self.gamma = kwargs.get('gamma', 0.99)
        self.lambda_ = kwargs.get('lambda_', 0.95)

        self.rho_bar = kwargs.get('rho_bar', 1.0)  # Importance sampling truncation
        self.c_bar = kwargs.get('c_bar', 1.0)  # Importance sampling clipping
        self.use_ppo = kwargs.get('use_ppo', False)
        self.clip_param = kwargs.get('clip_param', 0.2)

    def get_action(self, observation, evaluate=False):
        with torch.no_grad():
            action_prob, state_values = self.model(observation)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)
                return action.cpu().numpy(), None
            else:
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                return action.cpu().numpy(), {
                    "log_probs": log_prob.cpu().numpy(),
                }


    def process_trajectory(self, trajectory):
        obs = trajectory['observations'][:-1]  # 去掉最后一个终止状态
        next_obs = trajectory['observations'][1:]  # 对应 obs 的下一个状态
        actions = trajectory['actions']  # T+1
        old_log_probs = trajectory['log_probs'] # T+1
        rewards = trajectory['rewards']  # 长度本就为 T，无需修改
        dones = trajectory['dones']
        episode_len = len(rewards)

        with torch.no_grad():
            new_action_probs, new_values = self.model(torch.FloatTensor(np.array(trajectory['observations'])).to(self.model.device))  # 计算 obs 的策略概率 & value
            action_dist = Categorical(new_action_probs)
            new_log_probs = action_dist.log_prob(torch.from_numpy(np.array(actions)).to(self.model.device))
            new_log_probs = new_log_probs.detach().cpu().numpy()
            new_values = new_values.detach().cpu().numpy()
        # 计算重要性采样比率 (剪裁)
        rho = np.minimum(np.exp(new_log_probs - old_log_probs), self.rho_bar)
        c = np.minimum(np.exp(new_log_probs - old_log_probs), self.c_bar)  * self.lambda_

        vtrace = np.zeros(episode_len + 1, dtype=np.float32)
        vtrace[-1] = new_values[-1]
        advantages = np.zeros(episode_len, dtype=np.float32)
        acc = 0 # GAR形式累积因子
        for s in reversed(range(episode_len)):
            delta_V_s = rho[s] * (rewards[s] + self.gamma * new_values[s + 1] * (1.0 - dones[s]) - new_values[s])
            acc = delta_V_s + self.gamma * c[s] * acc
            vtrace[s] = new_values[s] + acc
            advantages[s] = rho[s] * (rewards[s] + self.gamma * vtrace[s + 1] * (1.0 - dones[s]) - new_values[s])
        transitions = list(zip(obs, actions[:-1], rewards, old_log_probs[:-1], vtrace[:-1], vtrace[1:], advantages))
        return transitions


    def train(self, data, weights=None):
        obs, actions, rewards, old_log_probs, vtrace, vtrace_nxt, advantages = data
        action_probs, new_values = self.model(obs)
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        # Vtrace advantage
        rho = torch.clip(torch.exp(new_log_probs.detach() - old_log_probs), max=self.rho_bar)
        advantages = rho * (rewards + self.gamma * vtrace_nxt - new_values.detach())
        if self.use_ppo:
            ratio = torch.exp(new_log_probs - old_log_probs)  # 计算概率比率
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = (-torch.min(surr1, surr2)).mean()
        else:
            actor_loss = (-new_log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(new_values, vtrace)
        td_errors = (vtrace - new_values.detach()).abs().cpu().numpy()
        if weights is not None:
            critic_loss = weights * critic_loss

        loss = actor_loss + critic_loss
        loss = loss.mean().unsqueeze(-1)

        self.model.backward(loss)
        self.model.step()
        self.train_step += 1

        return self.train_step, loss.detach().cpu().numpy(), td_errors
