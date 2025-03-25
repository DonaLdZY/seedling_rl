import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPO:
    def __init__(self, actor, critic, setting: dict):
        self.device = setting.get('device', torch.device('cpu'))
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.eval_mode = setting.get('eval_mode', False)
        if not self.eval_mode:
            self.train_step = 0
            self.save_name = setting.get('save_name', 'A2C')
            self.save_step = setting.get('save_step', 1000)
            self.clip_param = setting.get('clip_param', 0.2)
            try:
                self.actor_optimizer = setting['actor_optimizer']
                self.critic_optimizer = setting['critic_optimizer']
            except KeyError:
                raise KeyError('Both actor_optimizer and critic_optimizer are required')

    def save_model(self, file_name):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, file_name + ".pth")

    def load_model(self, file_name):
        checkpoint = torch.load(file_name + ".pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

    def train(self, data):
        if self.eval_mode:
            raise KeyError("can not train in eval mode")
        observations, actions, old_log_probs, returns, advantages = data

        observations = observations.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

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

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.train_step += 1
        if self.train_step % self.save_step == 0:
            self.save_model(self.save_name)

        return self.train_step, (actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy())

    def get_action(self, observation, evaluate=False):
        with (torch.no_grad()):
            observation_tensor = torch.FloatTensor(observation).to(self.device)
            action_prob = self.actor(observation_tensor)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)
                return action.cpu().numpy(), None
            else:
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                value = self.critic(observation_tensor)
                return action.cpu().numpy(), \
                    {
                        "log_probs":log_prob.cpu().numpy(),
                        "values":value.cpu().numpy()
                    }
