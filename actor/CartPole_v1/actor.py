import grpc
import threading
import pickle

from comm.service_pb2_grpc import LearnerStub
from comm.service_pb2 import Actions, Observations

import gymnasium as gym
import time
import random


class Actor(threading.Thread):
    def __init__(self, host, port):
        super(Actor, self).__init__()
        self.env = gym.make("CartPole-v1", render_mode="human")

        self.host = f"{host}:{port}"
        channel = grpc.insecure_channel(self.host, options=[('grpc.enable_http_proxy', 0)])  # 禁用代理
        self.stub = LearnerStub(channel)

    def run(self):
        while True:
            env_id = random.random()
            step_count = 0
            observation = self.env.reset(seed=int(time.time()))[0]
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
            while not (terminated or truncated):
                _observation = {
                    "env_id":env_id,
                    "step_count":step_count,
                    "observation":observation,
                    "reward":reward,
                    "terminated":terminated,
                    "truncated":truncated,
                    "info":info
                }
                _observation = pickle.dumps(_observation)
                response = self.stub.GetActions(Observations(observation=_observation))
                action = pickle.loads(response.actions)
                print(type(action))
                print(type(action[0]))
                print(action[0])
                observation, reward, terminated, truncated, info = self.env.step(action[0])