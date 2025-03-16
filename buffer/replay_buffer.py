import numpy as np
class ReplayBuffer:
    def __init__(self, capacity, startup:int = None):
        self.buffer=[]
        self.capacity = capacity
        self.startup = startup if startup is not None else capacity
        self.current_state={}

    def __len__(self):
        return len(self.buffer)
    def len(self):
        return self.__len__()

    def __sizeof__(self):
        return self.capacity
    def size(self):
        return self.__len__()

    def ready(self):
        return self.size() >= self.startup

    def store(self, observation, action, action_prob):
        if observation['step_count']>0:
            if len(self.buffer) == self.capacity:
                self.buffer.pop(0)

            # TODO : reward design
            self.buffer.append((
                self.current_state[observation['env_id']][0], # observation
                self.current_state[observation['env_id']][1], # action
                observation['reward'],
                observation['observation'], # next_action
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
