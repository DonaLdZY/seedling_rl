import grpc
from concurrent import futures
import threading
import pickle
import collections
import time
import torch
from torch import optim
import numpy as np

from comm.service_pb2_grpc import LearnerServicer, add_LearnerServicer_to_server
from comm.service_pb2 import Actions, Observations
from agent.DQN import DQN
from actor.CartPole_v1.network.QNet import Network
from actor.CartPole_v1.env_info import random_move
from buffer.replay_buffer import ReplayBuffer
class Learner(LearnerServicer):
    def __init__(self):
        self.data_count = 0
        self.save_name = 'test'
        model=Network()
        agent_args={
            'device':torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0'),
            'eval_mode':False,
            'optimizer':optim.Adam(
                model.parameters(),
                lr=0.0003,
                weight_decay=0.0005
            ),
            'random_move':random_move,
        }
        self.agent = DQN(Network(), agent_args)
        self.buffer = ReplayBuffer(capacity=1024)
        self.time_begin=time.time()
        self.training_thread = threading.Thread(target=self.train)
        self.training_thread.daemon = True  # 设置为守护线程
        self.training_thread.start()

    def GetActions(self, request, context):
        print("GetActions!!!!!!!!")
        observation = request.observation
        print("observation", type(observation))
        observation = pickle.loads(observation)  #

        # _observation = {"env_id": env_id,
        #                 "step_count": step_count,
        #                 "observation": observation,
        #                 "reward": reward,
        #                 "terminated": terminated,
        #                 "truncated": truncated,
        #                 "info": info}
        print("DONE!!!!!")
        action = None
        action_prob = None
        if not observation['terminated'] or observation['truncated']:
            print("GO!!!!!")
            print(observation['observation'])
            action, action_prob=self.agent.get_action(observation['observation'])
        print("DONE 2!!!!!")
        print(type(action))
        self.buffer.store(observation, action)
        self.data_count += 1
        print("DONE 3!!!!!")
        return Actions(actions=pickle.dumps([action]))

    def train(self):
        while True:
            if self.buffer.len() >= self.buffer.capacity:
                self.agent.train(self.buffer.sample(64))
                if self.agent.train_step % self.agent.sync_target_step == 0:
                    self.agent.save_model(self.save_name)
            else:
                time.sleep(10)

# 创建一个全局的Learner对象
global_learner = Learner()

def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_LearnerServicer_to_server(global_learner, server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    print('gRPC 服务端已开启，端口为' + str(port) + '...')
    server.wait_for_termination()

if __name__ == '__main__':
    thread = threading.Thread(target=serve, args=(50051,))
    thread.daemon = True
    thread.start()
    thread.join()