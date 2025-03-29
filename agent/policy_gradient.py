import torch
from torch.distributions import Categorical
from utils.create_optimizer import create_optimizer

class PolicyGradient:
    def __init__(self, network, **kwargs):
        self.device = kwargs.get('device', torch.device('cpu'))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.network = network.to(self.device)
        try:
            self.optimizer = create_optimizer(self.network.parameters(), kwargs['optimizer'],
                                              **kwargs.get('optimizer_args', {}))
        except KeyError:
            raise KeyError('optimizer is required')
        self.train_step = 0
        self.save_name = kwargs.get('save_name', 'PolicyGradient')
        self.save_step = kwargs.get('save_step', 1000)


    def save_model(self, file_name):
        torch.save(self.network.state_dict(), file_name + ".pth")

    def load_model(self, file_name):
        self.network.load_state_dict(torch.load(file_name + ".pth", map_location=self.device))


    def train(self, data, weights=None):
        observations, actions, rewards, next_observations, dones = data
        observations = observations.to(self.device)
        next_observations = next_observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        action_probs = self.network(observations)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step+=1
        if self.train_step % self.save_step == 0:
            self.save_model(self.save_name)
        return (self.train_step,
                loss.detach().cpu().numpy(),
                None)


    def get_action(self, observation, evaluate=False):
        with torch.no_grad():
            observation_tensor = torch.FloatTensor(observation).to(self.device)
            action_prob = self.network(observation_tensor)
            action_dist = Categorical(action_prob)
            if evaluate:
                action = torch.argmax(action_prob, dim=-1)  # 选取概率最高的动作
            else:
                action = action_dist.sample()
        return action.cpu().numpy(), None

