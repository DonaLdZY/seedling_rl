import numpy as np
from multiprocessing import Manager

import time

class IncompleteTrajectoriesBuffer:
    def __init__(self, output_queue):
        self.manager = Manager()
        self.current_trajectories = {}
        self.output_queue = output_queue

        self.count = 0
        self.score = []
        self.time = time.time()

    def store(self, observations, actions, extra_dist):
        if extra_dist is None:
            extra_dist = {}
        env_ids, next_obs, rewards, terminated, truncated = observations
        batch_size = len(env_ids)
        for i in range(batch_size):
            env_id = env_ids[i]
            if env_id not in self.current_trajectories.keys():
                self.current_trajectories[env_id] = {
                    'observations': [next_obs[i]],
                    'actions': [actions[i]],
                    'rewards': [],
                }
                for key, value in extra_dist.items():
                    self.current_trajectories[env_id][key] = [value[i]]
            else:
                self.current_trajectories[env_id]['observations'].append(next_obs[i])
                self.current_trajectories[env_id]['actions'].append(actions[i])
                self.current_trajectories[env_id]['rewards'].append(rewards[i])
                for key, value in extra_dist.items():
                    self.current_trajectories[env_id][key].append(value[i])

            if terminated[i] or truncated[i]:
                complete_trajectory = self.current_trajectories.pop(env_id)
                self.process_completed_trajectory(env_id, complete_trajectory)

    def process_completed_trajectory(self, env_id, complete_trajectory):
        self.logging(env_id, complete_trajectory)
        self.output_queue.put(complete_trajectory)

    def logging(self, env_id, complete_trajectory):
        self.score.append(np.sum(np.array(complete_trajectory['rewards'])))
        now = time.time()
        if now - self.time >= 1:
            print(f'log | average score | : ', np.mean(np.array(self.score)))
            self.time = now
            self.score = []
