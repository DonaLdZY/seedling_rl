import numpy as np

import gymnasium as gym
env = gym.make('LunarLander-v3')
action_space = env.action_space
observation_space = env.observation_space
n_observation = 8
n_action = 4
def random_move():
    return env.action_space.sample()

if __name__ == '__main__':
    print(action_space)
    print(observation_space)