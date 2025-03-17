import grpc
from concurrent import futures
import threading
import pickle
import time

from comm.service_pb2_grpc import LearnerServicer, add_LearnerServicer_to_server
from comm.service_pb2 import Request, Response

def run_learner(agent, buffer, port="50051"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_LearnerServicer_to_server(Learner(agent=agent, buffer=buffer), server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    print('gRPC 服务端已开启，端口为' + str(port) + '...')
    server.wait_for_termination()

class Learner(LearnerServicer):
    def __init__(self, agent, buffer):
        self.agent = agent
        self.buffer = buffer
        self.training_thread = threading.Thread(target=self.train)
        self.training_thread.daemon = True  # 设置为守护线程
        self.training_thread.start()

    def get_action(self, observation):
        # observation = {
        #     "env_id": env_id,
        #     "observation": observation,
        #     "reward": reward,
        #     "terminated": terminated,
        #     "truncated": truncated,
        # }
        action, action_prob = self.agent.get_action(observation['observation'])
        self.buffer.store(observation, action, action_prob)
        return action

    def train(self):
        while True:
            if self.buffer.ready():
                train_step, loss = self.agent.train(self.buffer.sample(64))
                print("train!!!", train_step, loss.item())
            else:
                time.sleep(10)

    def getAction(self, request, context):
        observation = pickle.loads(request.observation)
        action = self.get_action(observation)
        _action = pickle.dumps(action)
        return Response(action=_action)

    def getActionStream(self, request_iterator, context):
        for request in request_iterator:
            observation = pickle.loads(request.observation)
            _action = pickle.dumps(self.get_action(observation))
            yield Response(action=_action)



