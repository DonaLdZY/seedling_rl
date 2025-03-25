import torch
from torch.distributions import Categorical

class PolicyGradient:
    def __init__(self, network, setting:dict):
        self.device = setting.get('device', torch.device('cpu'))
        self.network = network.to(self.device)
        self.eval_mode = setting.get('eval_mode', False)
        if not self.eval_mode:
            self.train_step = 0
            self.save_name = setting.get('save_name', 'PolicyGradient')
            self.save_step = setting.get('save_step', 1000)

            try:
                self.optimizer = setting['optimizer']
            except Exception:
                raise KeyError('optimizer is required')
        pass

    def save_model(self, file_name):
        torch.save(self.network.state_dict(), file_name + ".pth")

    def load_model(self, file_name):
        self.network.load_state_dict(torch.load(file_name + ".pth", map_location=self.device))

    def train(self, data):
        if self.eval_mode:
            raise KeyError("can not train in eval mode")

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
        return self.train_step, loss.detach().numpy()

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

