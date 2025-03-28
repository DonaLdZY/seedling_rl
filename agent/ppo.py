import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.create_optimizer import create_optimizer
from utils.create_scheduler import create_scheduler

class PPO:
    def __init__(self, actor, critic, **kwargs):
        self.device = kwargs.get('device', torch.device('cpu'))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        try:
            self.actor_optimizer = create_optimizer(self.actor.parameters(), kwargs['actor_optimizer'])
            self.critic_optimizer = create_optimizer(self.critic.parameters(), kwargs['critic_optimizer'])
        except KeyError:
            raise KeyError('Both actor_optimizer and critic_optimizer are required')
        self.actor_scheduler = create_scheduler(self.actor_optimizer,
                                                kwargs['actor_scheduler']) if 'actor_scheduler' in kwargs else None
        self.critic_scheduler = create_scheduler(self.critic_optimizer,
                                                 kwargs['critic_scheduler']) if 'critic_scheduler' in kwargs else None
        self.train_step = 0
        self.save_name = kwargs.get('save_name', 'PPO')
        self.save_step = kwargs.get('save_step', 1000)
        self.clip_param = kwargs.get('clip_param', 0.2)

    def save_model(self, file_name):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, file_name + ".pth")

    def load_model(self, file_name):
        checkpoint = torch.load(file_name + ".pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

    def train(self, data, weights=None):
        observations, actions, old_log_probs, returns, advantages = data

        action_probs = self.actor(observations)
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        # 计算概率比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        # PPO 剪切目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        # 价值网络损失
        values = self.critic(observations)
        critic_loss = F.mse_loss(values, returns)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if weights is not None:
            critic_loss = (weights * critic_loss).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        self.train_step += 1
        if self.train_step % self.save_step == 0:
            self.save_model(self.save_name)

        td_errors = torch.abs(returns - values.detach()).cpu().numpy()

        return (self.train_step,
                (actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()),
                td_errors)


    def get_action(self, observation, evaluate=False):
        with (torch.no_grad()):
            action_prob = self.actor(observation)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)
                return action.cpu().numpy(), None
            else:
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                value = self.critic(observation)
                return action.cpu().numpy(), \
                    {
                        "log_probs":log_prob.cpu().numpy(),
                        "values":value.cpu().numpy()
                    }
