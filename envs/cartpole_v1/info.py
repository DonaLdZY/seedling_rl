import numpy as np

import gymnasium as gym
env = gym.make('CartPole-v1')
print("ok")
action_space = env.action_space
observation_space = env.observation_space
n_observation = 4
n_action = 2
def random_move():
    return env.action_space.sample()