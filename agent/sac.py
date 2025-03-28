import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.create_optimizer import create_optimizer
from torch import optim
import numpy as np
import copy

class SAC:
    def __init__(self, actor, critic1, critic2, **kwargs):
        self.device = kwargs.get('device', torch.device('cpu'))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        try:
            self.actor_optimizer = create_optimizer(self.actor.parameters(), kwargs['actor_optimizer'],
                                                    **kwargs.get('actor_optimizer_args', {}))
            self.critic_1_optimizer = create_optimizer(self.critic1.parameters(), kwargs['critic_1_optimizer'],
                                                    **kwargs.get('critic_1_optimizer_args', {}))
            self.critic_2_optimizer = create_optimizer(self.critic2.parameters(), kwargs['critic_2_optimizer'],
                                                    **kwargs.get('critic_2_optimizer_args', {}))
        except KeyError:
            raise KeyError('actor_optimizer, critic_1_optimizer, critic_2_optimizer are required')
        self.train_step = 0
        self.save_name = kwargs.get('save_name', 'SAC')
        self.save_step = kwargs.get('save_step', 1000)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)

        # 温度参数，用于控制熵奖励，若设置自动调节，则需要优化 log_alpha
        self.target_entropy = kwargs.get('target_entropy', -1.0)  # 一般设为 -|A|
        # 初始化一个标量参数，取指数即为温度 alpha
        self.log_alpha = torch.tensor(np.log(kwargs.get('alpha', 1)),
                                      dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_lr = kwargs.get('alpha_lr', 1e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.alpha = self.log_alpha.exp()

        # 构造 target critic 网络，初始时直接复制 critic 参数
        self.critic1_target = copy.deepcopy(critic1)
        self.critic2_target = copy.deepcopy(critic2)
        self.critic1_target.to(self.device)
        self.critic2_target.to(self.device)
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

    def save_model(self, file_name):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha
        }, file_name + ".pth")

    def load_model(self, file_name):
        checkpoint = torch.load(file_name + ".pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()

    def train(self, data):
        observations, actions, rewards, next_obs, dones = data
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        # 计算 target Q 值
        with torch.no_grad():
            next_action_probs = self.actor(next_obs)
            next_q1_values = self.critic1(next_obs)
            next_q2_values = self.critic2(next_obs)
            min_next_q_values = torch.min(next_q1_values, next_q2_values)
            next_v = (next_action_probs * (min_next_q_values - self.alpha * torch.log(next_action_probs + 1e-8))).sum(
                dim=1)
            target_q = rewards + self.gamma * (1 - dones) * next_v

        # 计算当前 Q 值预测
        current_q1_values = self.critic1(observations)
        current_q2_values = self.critic2(observations)
        current_q1 = current_q1_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = current_q2_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # 更新 actor 网络，注意 actor 网络需要利用 critic 网络梯度来调整策略
        new_action_probs = self.actor(observations)
        new_q1_values = self.critic1(observations)
        new_q2_values = self.critic2(observations)
        min_new_q_values = torch.min(new_q1_values, new_q2_values)
        actor_loss = (new_action_probs * (self.alpha * torch.log(new_action_probs + 1e-8) - min_new_q_values)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = - (self.log_alpha * (
                    new_action_probs * (torch.log(new_action_probs + 1e-8) + self.target_entropy)).sum(
            dim=1).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 软更新 target critic 网络参数
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        self.train_step += 1
        if self.train_step % self.save_step == 0:
            self.save_model(self.save_name)

        td_error1 = (target_q - current_q1).detach().abs().cpu().numpy()
        td_error2 = (target_q - current_q2).detach().abs().cpu().numpy()
        td_errors = np.maximum(td_error1, td_error2)

        return (self.train_step,
                (actor_loss.item(), critic_1_loss.item(), critic_2_loss.item()),
                td_errors)

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action(self, observation, evaluate=False):
        with torch.no_grad():
            observation_tensor = torch.FloatTensor(observation).to(self.device)
            action_prob = self.actor(observation_tensor)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)
            else:
                action = action_dist.sample()
            return action.cpu().numpy(), None

