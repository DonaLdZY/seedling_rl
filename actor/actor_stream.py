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

        self.obs_ready = threading.Event()
        self.obs_ready.set()

        self.count = 0
        self.record_time = time.time()

    def request_stream(self):
        while True:
            self.obs_ready.wait()
            _observation = pickle.dumps(self.observation)
            self.obs_ready.clear()
            yield Request(observation=_observation)

    def run(self):
        while True:
            responses = self.stub.getActionStream(self.request_stream())
            for response in responses:
                action = pickle.loads(response.action)
                self.observation = self.env.step(action)
                self.obs_ready.set()
                self.logging()

    def logging(self):
        self.count += 1
        now = time.time()
        if now - self.record_time > 5.0:
            print(self.count / (now - self.record_time))
            self.count = 0
            self.record_time = now

