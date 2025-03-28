import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class A2C:
    def __init__(self, actor, critic, setting: dict):
        self.device = setting.get('device', torch.device('cpu'))
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.eval_mode = setting.get('eval_mode', False)
        if not self.eval_mode:
            self.train_step = 0
            self.save_name = setting.get('save_name', 'A2C')
            self.save_step = setting.get('save_step', 1000)
            self.gamma = setting.get('gamma', 0.99)
            try:
                self.actor_optimizer = setting['actor_optimizer']
                self.critic_optimizer = setting['critic_optimizer']
            except KeyError:
                raise KeyError('Both actor_optimizer and critic_optimizer are required')

    def save_model(self, file_name):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, file_name + ".pth")

    def load_model(self, file_name):
        checkpoint = torch.load(file_name + ".pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

    def train(self, data):
        if self.eval_mode:
            raise KeyError("can not train in eval mode")

        observations, actions, rewards, next_observations, dones = data

        observations = observations.to(self.device)
        next_observations = next_observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        action_probs = self.actor(observations)
        state_values = self.critic(observations)
        next_state_values = self.critic(next_observations)

        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        target_values = rewards + self.gamma * next_state_values * (1 - dones)

        advantages = target_values - state_values
        actor_loss = (-log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(state_values, target_values.detach())

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
        with torch.no_grad():
            observation_tensor = torch.FloatTensor(observation).to(self.device)
            action_prob = self.actor(observation_tensor)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)
            else:
                action = action_dist.sample()
        return action.cpu().numpy(), None
