import grpc
import threading
import pickle

from comm.service_pb2_grpc import LearnerStub
from comm.service_pb2 import Request, Response

class Actor(threading.Thread):
    def __init__(self, env, host, port, ):
        super(Actor, self).__init__()
        self.env = env

        self.host = f"{host}:{port}"
        channel = grpc.insecure_channel(self.host, options=[('grpc.enable_http_proxy', 0)])  # 禁用代理
        self.stub = LearnerStub(channel)

    def run(self):
        observation = self.env.reset()
        while True:
            _observation = pickle.dumps(observation)
            response = self.stub.getAction(Request(observation=_observation))
            action = pickle.loads(response.action)
            observation = self.env.step(action)
