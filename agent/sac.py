import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.create_optimizer import create_optimizer
from utils.create_scheduler import create_scheduler
from torch import optim
import numpy as np
import copy

class SAC:
    def __init__(self, model, **kwargs):
        self.model = model
        self.train_step = 0
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)

        # 温度参数，用于控制熵奖励，若设置自动调节，则需要优化 log_alpha
        self.target_entropy = kwargs.get('target_entropy', -1.0)  # 一般设为 -|A|
        # 初始化一个标量参数，取指数即为温度 alpha
        self.log_alpha = torch.tensor(np.log(kwargs.get('alpha', 1)),
                                      dtype=torch.float32, requires_grad=True, device=self.model.device)
        self.alpha_lr = kwargs.get('alpha_lr', 1e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.alpha = self.log_alpha.exp()

        # target critic network
        self.target_network = copy.deepcopy(self.model.module)
        self.target_network.to(self.model.device)
        self.target_network.eval()


    def get_action(self, observation, evaluate=False):
        with torch.no_grad():
            action_prob, q1_value, q2_value = self.model(observation)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)
            else:
                action = action_dist.sample()
            return action.cpu().numpy(), None


    def process_trajectory(self, trajectory):
        obs = trajectory['observations'][:-1]  # 去掉最后一个终止状态
        next_obs = trajectory['observations'][1:]  # 对应 obs 的下一个状态
        actions = trajectory['actions'][:-1]
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        transitions = list(zip(obs, actions, rewards, next_obs, dones))
        return transitions


    def train(self, data, weights=None):
        observations, actions, rewards, next_obs, dones = data

        # 计算 target Q 值
        with torch.no_grad():
            next_action_probs, next_q1_values, next_q2_values = self.model(next_obs)
            min_next_q_values = torch.min(next_q1_values, next_q2_values)
            next_v = (next_action_probs * (min_next_q_values - self.alpha * torch.log(next_action_probs.clamp(min=1e-8)))).sum(dim=1)
            target_q = rewards + self.gamma * (1 - dones) * next_v
        # 更新 actor 网络，注意 actor 网络需要利用 critic 网络梯度来调整策略
        action_probs, q1_values, q2_values = self.model(observations)
        q1 = q1_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = q2_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        td_error1 = (target_q - q1).detach().abs().cpu().numpy()
        td_error2 = (target_q - q2).detach().abs().cpu().numpy()
        td_errors = np.maximum(td_error1, td_error2)
        min_q_values = torch.min(q1_values, q2_values)

        actor_loss = (action_probs * (self.alpha * torch.log(action_probs + 1e-8) - min_q_values.detach())).sum(dim=1).mean()
        loss = q1_loss + q2_loss + actor_loss
        loss = loss.mean().unsqueeze(-1)

        self.model.backward(loss)
        self.model.step()
        self.train_step += 1

        alpha_loss = - (self.log_alpha * ((action_probs * (torch.log(action_probs.clamp(min=1e-8)) + self.target_entropy)).sum(dim=1).detach())).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self._soft_update(self.model.module, self.target_network)

        return (self.train_step,
                loss.detach().cpu().numpy(),
                td_errors)

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


