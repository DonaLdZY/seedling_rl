import numpy as np
from multiprocessing import Manager

import time

class IncompleteTrajectoriesBuffer:
    def __init__(self, output_queue):
        self.manager = Manager()
        self.current_trajectories = {}
        self.output_queue = output_queue

        self.count = 0
        self.time = time.time()

    def store(self, observations, actions, action_probs):
        env_ids, next_obs, rewards, terminated, truncated = observations
        batch_size = len(env_ids)
        if action_probs is None:
            action_probs = np.array([None] * batch_size)
        for i in range(batch_size):
            env_id = env_ids[i]
            if env_id not in self.current_trajectories.keys():
                self.current_trajectories[env_id] = {
                    'observations': [next_obs[i]],
                    'rewards': [],
                    'actions': [actions[i]],
                    'action_probs': [],
                }
            else:
                self.current_trajectories[env_id]['observations'].append(next_obs[i])
                self.current_trajectories[env_id]['actions'].append(actions[i])
                self.current_trajectories[env_id]['action_probs'].append(action_probs[i])
                self.current_trajectories[env_id]['rewards'].append(rewards[i])

            if terminated[i] or truncated[i]:
                complete_trajectory = self.current_trajectories.pop(env_id)
                self.process_completed_trajectory(env_id, complete_trajectory)

    def process_completed_trajectory(self, env_id, complete_trajectory):
        self.logging(env_id, complete_trajectory)
        self.output_queue.put(complete_trajectory)

    def logging(self, env_id, complete_trajectory):
        now = time.time()
        if now - self.time >= 5:
            print(f'log | action | env[ {env_id} ] : ', np.sum(np.array(complete_trajectory['rewards'])))
            self.time = now
