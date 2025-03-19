import gymnasium as gym
import numpy as np
import time
class Env:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.env = gym.make_vec('CartPole-v1', num_envs=num_envs)

        self.env_id = None
        self.observation = None
        self.reward = None
        self.terminated = None
        self.truncated = None
        self.reset()

    def reset(self):
        self.env_id = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=self.num_envs, dtype=np.int32)
        self.observation, _ = self.env.reset(seed=int(time.time()))
        self.reward = np.zeros(self.num_envs, dtype=float)
        self.terminated = np.zeros(self.num_envs, dtype=bool)
        self.truncated = np.zeros(self.num_envs, dtype=bool)
        return self._observation()

    def step(self, action):
        self.observation, self.reward, self.terminated, self.truncated, _ = self.env.step(action)
        return self._observation()

    def _observation(self):
        return (
            self.env_id,
            self.observation,
            self.reward,
            self.terminated,
            self.truncated,
        )
