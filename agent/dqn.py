import numpy as np
import torch
import torch.nn.functional as F
import threading
import copy
import random

class DQN:
    def __init__(self, model, **kwargs):
        self.model = model
        self.train_step = 0
        self.lock = threading.Lock()
        self.gamma = kwargs.get('gamma', 0.99)
        # dqn target_network
        self.target_network = copy.deepcopy(self.model.module)
        self.target_network.to(self.model.device)
        self.target_network.eval()
        self.sync_freq = kwargs.get('sync_freq', 1000)
        # epsilon_greedy
        self.epsilon_max = max(0.0, min(kwargs.get('epsilon_max', 0.8), 1.0))
        self.epsilon_min = max(0.0, min(kwargs.get('epsilon_min', 0.05), self.epsilon_max))
        self.epsilon_decay = max(0.0, min(kwargs.get('epsilon_decay', 0.9995), 1.0))
        self.epsilon = self.epsilon_max

        self.double_dqn = kwargs.get('double_dqn', False)


    def get_action(self, observation, evaluate=False, epsilon=None):
        epsilon = epsilon if epsilon else self.epsilon
        with torch.no_grad():
            if evaluate or (not np.random.uniform()<=epsilon):
                with self.lock:
                    action_value = self.model(observation)
                action = torch.argmax(action_value, dim=-1).cpu().numpy()
            else :
                action = np.array([self.random_move() for _ in range(observation.shape[0])])
        return action, None
    def random_move(self):
        return random.randint(0, self.model.module.output_dim - 1)


    def process_trajectory(self, trajectory):
        obs = trajectory['observations'][:-1]  # 去掉最后一个终止状态
        next_obs = trajectory['observations'][1:]  # 对应 obs 的下一个状态
        actions = trajectory['actions'][:-1]
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        transitions = list(zip(obs, actions, rewards, next_obs, dones))
        return transitions


    def train(self, data, weights=None):
        obs, actions, rewards, nxt_obs, dones = data

        if self.train_step % self.sync_freq == 0:
            self.target_network.load_state_dict(self.model.module.state_dict())
        pred = self.model(obs)
        q_eval = pred.gather(1, actions.unsqueeze(1)).squeeze(1)
        if self.double_dqn:
            next_actions = self.model(nxt_obs).argmax(dim=1, keepdim=True)
            q_next = self.target_network(nxt_obs).gather(1, next_actions).squeeze(1)
        else:
            q_next = self.target_network(nxt_obs).max(1)[0]
        q_target = rewards + self.gamma * q_next * (1 - dones)
        td_errors = (q_target - q_eval).detach().abs().cpu().numpy()
        loss = F.mse_loss(q_eval, q_target.detach())
        if weights is not None:
            loss = weights * loss
        loss = loss.mean().unsqueeze(-1)

        self.model.backward(loss)
        self.model.step()
        self.train_step+=1

        return (self.train_step,
                loss.detach().cpu().numpy(),
                td_errors)




