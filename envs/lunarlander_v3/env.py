import gymnasium as gym
import numpy as np
import time
class Env:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.env = gym.make_vec('LunarLander-v3', num_envs=num_envs)

        self.env_id = None
        self.observation = None
        self.reward = None
        self.terminated = None
        self.truncated = None
        self.reset()

    def reset(self):
        self.env_id = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=self.num_envs, dtype=np.int32)
        self.observation, _ = self.env.reset(seed=int(time.time()))
        self.reward = np.zeros(self.num_envs, dtype=np.float32)
        self.terminated = np.zeros(self.num_envs, dtype=np.bool_)
        self.truncated = np.zeros(self.num_envs, dtype=np.bool_)
        return self.get_observation()

    def step(self, action):
        self.observation, self.reward, self.terminated, self.truncated, _ = self.env.step(action)
        return self.get_observation()

    def get_observation(self):
        return (
            self.env_id,
            self.observation.astype(np.float32),
            self.reward.astype(np.float32),
            self.terminated,
            self.truncated,
        )

if __name__ == '__main__':
    env = Env(1)
    def random_move():
        return env.env.action_space.sample()
    eid, obs, reward, ter, tru = env.get_observation()
    print(type(eid[0]))
    print(type(obs[0][0]))
    print(type(reward[0]))
    while not ter[0] and not tru[0]:
        eid, obs, reward, ter, tru = env.step(random_move())
        print(type(eid[0]))
        print(type(obs[0][0]))
        print(type(reward[0]))