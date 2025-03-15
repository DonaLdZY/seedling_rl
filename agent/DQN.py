import numpy as np
import torch
from torch import nn
import copy
class DQN:
    def __init__(self, model, setting:dict):
        # 模型model
        self.eval_model = model
        self.device = setting.get('device', 'cpu')
        self.eval_mode = setting.get('eval_mode', False)
        if not self.eval_mode:
            self.train_step = 0
            if 'optimizer' in setting:
                self.optimizer = setting['optimizer']
            else:
                raise KeyError('optimizer is required')

            self.target_model = copy.deepcopy(model)
            self.sync_target_step = setting.get('sync_target_step',100)
            self.gamma = setting.get('gamma', 0.99)
            self.criterion = setting.get('criterion', nn.MSELoss())
            self.epsilon_max = max(0.0, min(setting.get('epsilon_max',0.8), 1.0))
            self.epsilon_min = max(0.0, min(setting.get('epsilon_min',0.05), self.epsilon_max))
            self.epsilon_decay = max(0.0, min(setting.get('epsilon_decay',0.999), 1.0))
            self.epsilon = self.epsilon_max

            # random得在外面定义
            if "random_move" in setting:
                self.random_move = setting['random_move']
            else:
                if self.epsilon_max > 0.0:
                    raise KeyError("random_move() is required")
                self.random_move = None
        pass

    def save_model(self, file_name):
        torch.save(self.eval_model.state_dict(), file_name + ".pth")

    def load_model(self, file_name):
        self.eval_model.load_state_dict(torch.load(file_name + ".pth", map_location=self.device))
        self.sync_target()

    def sync_target(self):
        self.target_model.load_state_dict(self.eval_model.state_dict())

    def train(self, data):
        if self.eval_mode:
            raise KeyError("can not train in eval mode")
        if self.train_step % self.sync_target_step == 0:
            self.sync_target()
        observations, actions, rewards, next_observations, dones = data
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        q_eval = self.eval_model(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next = self.target_model(next_observations).max(1)[0].detach()
        q_target = rewards + self.gamma * q_next * (1 - dones)
        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step+=1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return self.train_step, loss

    def get_action(self, observation, evaluate=False):
        print(observation)
        observation = np.array(observation)
        if evaluate or self.eval_mode:
            with self.eval_model.no_grad():
                action_value = self.eval_model(observation).detach()
                action = torch.argmax(action_value, dim=-1)
        elif np.random.uniform()<=self.epsilon:
            action = self.random_move()
            action_value = None
        else:
            print(observation.shape)
            action_value = self.eval_model(observation).detach().cpu().numpy()
            print(action_value.shape)
            action = np.argmax(action_value)
        print(action, action_value)
        return action, action_value

