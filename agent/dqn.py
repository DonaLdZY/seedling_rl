import numpy as np
import torch
import torch.nn.functional as F
from utils.create_optimizer import create_optimizer
from utils.create_scheduler import create_scheduler
import copy
import random

class DQN:
    def __init__(self, network, **kwargs):
        self.device = kwargs.get('device', torch.device('cpu'))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.network = network.to(self.device)
        try:
            self.optimizer = create_optimizer(self.network.parameters(), kwargs['optimizer'])
        except KeyError:
            raise KeyError('optimizer is required')
        self.scheduler = create_scheduler(self.optimizer, kwargs['scheduler']) if 'scheduler' in kwargs else None

        self.train_step = 0
        self.save_name = kwargs.get('save_name', 'DQN')
        self.save_step = kwargs.get('save_step', 1000)
        self.gamma = kwargs.get('gamma', 0.99)

        self.double_dqn = kwargs.get('double_dqn', False)
        self.epsilon_max = max(0.0, min(kwargs.get('epsilon_max', 0.8), 1.0))
        self.epsilon_min = max(0.0, min(kwargs.get('epsilon_min', 0.05), self.epsilon_max))
        self.epsilon_decay = max(0.0, min(kwargs.get('epsilon_decay', 0.9995), 1.0))
        self.epsilon = self.epsilon_max
        self.network_target = copy.deepcopy(network)
        self.sync_target_step = kwargs.get('sync_target_step', 100)


    def save_model(self, file_name):
        torch.save(self.network.state_dict(), file_name + ".pth")

    def load_model(self, file_name):
        self.network.load_state_dict(torch.load(file_name + ".pth", map_location=self.device))
        self.sync_target()


    def train(self, data, weights=None):
        observations, actions, rewards, next_observations, dones = data

        if self.train_step % self.sync_target_step == 0:
            self.sync_target()

        pred = self.network(observations)
        q_eval = pred.gather(1, actions.unsqueeze(1)).squeeze(1)
        if self.double_dqn:
            next_actions = self.network(next_observations).argmax(dim=1, keepdim=True)
            q_next = self.network_target(next_observations).gather(1, next_actions).squeeze(1)
        else:
            q_next = self.network_target(next_observations).max(1)[0]
        q_target = rewards + self.gamma * q_next * (1 - dones)
        loss = F.mse_loss(q_eval, q_target.detach())

        if weights is not None:
            loss = (weights * loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.train_step+=1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if self.train_step % self.save_step == 0:
            self.save_model(self.save_name)

        td_errors = (q_target - q_eval).detach().abs().cpu().numpy()

        return (self.train_step,
                loss.detach().cpu().numpy(),
                td_errors)

    def sync_target(self):
        self.network_target.load_state_dict(self.network.state_dict())


    def get_action(self, observation, evaluate=False, epsilon=None):
        epsilon = epsilon if epsilon else self.epsilon
        with torch.no_grad():
            if evaluate or (not np.random.uniform()<=epsilon):
                action_value = self.network(observation)
                action = torch.argmax(action_value, dim=-1).cpu().numpy()
            else :
                action = np.array([self.random_move() for _ in range(observation.shape[0])])
        return action, None

    def random_move(self):
        return random.randint(0, self.network.output_dim - 1)

