import time

import numpy as np
from collections import deque
import random

class FIFOBuffer:
    def __init__(self, capacity, startup: int = None, target_fn = None):
        self._last_report_time = time.time()
        self._insertion_counter = 0
        self.capacity = capacity
        self.startup = startup if startup is not None else capacity
        self.buffer = deque(maxlen=capacity)
        self.target_fn = target_fn

    def store(self, trajectory):

        obs = trajectory['observations']
        episode_len = len(obs)-1
        next_obs = obs[1:]
        obs = obs[:-1]
        actions = trajectory['actions']
        actions = actions[:-1]
        rewards = trajectory['rewards']
        action_probs = trajectory['action_probs']
        dones = np.zeros(episode_len, dtype=np.float32)
        dones[-1] = 1.0

        targets = np.zeros(episode_len, dtype=np.float32) if self.target_fn is None else self.target_fn(trajectory)

        transitions = list(zip(obs, actions, rewards, next_obs, dones, targets))

        self.logging(len(transitions))

        self.buffer.extend(transitions)

    def sample(self, batch_size):
        index = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in index]
        observations, actions, rewards, next_observations, dones, targets = zip(*batch)
        return (np.array(observations),
                np.array(actions),
                np.array(rewards),
                np.array(next_observations),
                np.array(dones),
                np.array(targets))
    def ready(self):
        return len(self.buffer) >= self.startup

    def logging(self, count):
        self._insertion_counter += count
        now = time.time()
        if now - self._last_report_time >= 5:
            speed = self._insertion_counter / (now - self._last_report_time)
            print(f"log | buffer | 存入速度: {speed:.2f} items/s")
            self._insertion_counter = 0
            self._last_report_time = now
