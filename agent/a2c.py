import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.create_optimizer import create_optimizer

class A2C:
    def __init__(self, actor, critic, **kwargs):
        self.device = kwargs.get('device', torch.device('cpu'))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        try:
            self.actor_optimizer = create_optimizer(self.actor.parameters(), kwargs['actor_optimizer'],
                                                    **kwargs.get('actor_optimizer_args', {}))
            self.critic_optimizer = create_optimizer(self.critic.parameters(), kwargs['critic_optimizer'],
                                                     **kwargs.get('critic_optimizer_args', {}))
        except KeyError:
            raise KeyError('Both actor_optimizer and critic_optimizer are required')
        self.train_step = 0
        self.save_name = kwargs.get('save_name', 'A2C')
        self.save_step = kwargs.get('save_step', 1000)
        self.gamma = kwargs.get('gamma', 0.99)


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

        td_errors = advantages

        return (self.train_step,
                (actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()),
                td_errors)


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
