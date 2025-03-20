import grpc
import threading
import pickle
from comm.service_pb2_grpc import LearnerStub
from comm.service_pb2 import Request, Response

import time

class Actor(threading.Thread):
    def __init__(self, env, host, port):
        super(Actor, self).__init__()
        self.env = env
        self.observation = self.env.reset()

        self.host = f"{host}:{port}"
        channel = grpc.insecure_channel(self.host, options=[('grpc.enable_http_proxy', 0)])  # 禁用代理
        self.stub = LearnerStub(channel)

        self.count = 0
        self.record_time = time.time()

    def run(self):
        while True:
            _observation = pickle.dumps(self.observation)
            response = self.stub.getAction(Request(observation=_observation))
            action = pickle.loads(response.action)
            self.observation = self.env.step(action)
            self.logging()

    def logging(self):
        self.count += 1
        now = time.time()
        if now - self.record_time > 5.0:
            print(self.count / (now - self.record_time))
            self.count = 0
            self.record_time = now
