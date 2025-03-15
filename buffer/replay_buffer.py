import numpy as np
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=[]
        self.capacity=capacity
        self.current_state={}
    def __len__(self):
        return len(self.buffer)
    def len(self):
        return len(self.buffer)
    def store(self, observation, action):
        if observation['step_count']>0:
            if len(self.buffer) == self.capacity:
                self.buffer.pop(0)
            self.buffer.append((
                self.current_state[observation['env_id']][0],
                self.current_state[observation['env_id']][1],
                observation['reward'],
                observation['observation'],
                observation['terminated'] or observation['truncated'],
            ))

        if observation['terminated'] or observation['truncated']:
            self.current_state.pop(observation['env_id'])
        else:
            self.current_state[observation['env_id']] = (observation['observation'], action)

    def sample(self,n):
        index = np.random.choice(len(self.buffer), n)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)

    def clean(self):
        self.buffer.clear()
