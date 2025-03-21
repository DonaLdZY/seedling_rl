import grpc
from concurrent import futures
import threading
import pickle

from comm.service_pb2_grpc import LearnerServicer, add_LearnerServicer_to_server
from comm.service_pb2 import Request, Response
from learner.incomplete_trajectories_buffer import IncompleteTrajectoriesBuffer as Incomplete

from queue import Queue
import time

def run_learner(model, buffer, port="50051"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_LearnerServicer_to_server(Learner(model=model, buffer=buffer), server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    print('gRPC 服务端已开启，端口为' + str(port) + '...')
    server.wait_for_termination()

class Learner(LearnerServicer):
    def __init__(self, model, buffer):
        self.model = model
        self.complete_queue = Queue()
        self.incomplete = Incomplete(output_queue=self.complete_queue)
        self.buffer = buffer

        self.training_thread = threading.Thread(target=self._training_thread, daemon=True)
        self.training_thread.start()

        self.data_prefetching_thread = threading.Thread(target=self._data_prefetching_thread, daemon=True)
        self.data_prefetching_thread.start()

        self.action_time = time.time()
        self.action_count = 0
        self.train_time = time.time()
        self.train_count = 0
        self.train_loss = []


    def get_action(self, observation):
        action, action_prob = self.model.get_action(observation[1])
        self.incomplete.store(observation, action, action_prob)

        self.logging_action(observation[0].shape[0])


        return action

    def _data_prefetching_thread(self):
        print("data prefetching thread start...")
        while True:
            complete_trajectory = self.complete_queue.get()
            self.buffer.store(complete_trajectory)

    def _training_thread(self):
        print("training thread start...")
        while True:
            if self.buffer.ready():
                batch = self.buffer.sample(64)
                train_step, loss = self.model.train(batch)
                self.logging_train(batch[0].shape[0], loss)
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
            action = self.get_action(observation)
            _action = pickle.dumps(action)
            yield Response(action=_action)

    def logging_action(self, action_count):
        self.action_count += action_count
        now=time.time()
        if now - self.action_time >= 5:
            speed = self.action_count / (now - self.action_time)
            print(f"log | action thread | 吞吐速度: {speed:.2f} actions/s")
            self.action_time = now
            self.action_count = 0


    def logging_train(self, train_step, loss):
        self.train_count += train_step
        self.train_loss.append(loss.detach().numpy())
        now = time.time()
        if now - self.train_time >= 5:
            import numpy as np
            speed = self.train_count / (now - self.train_time)
            print(f"log | train thread | 训练速度 : {speed:.2f} steps/s")
            print("log | train thread | 平均loss :", np.mean(np.array(self.train_loss)))
            self.train_time = now
            self.train_loss = []
            self.train_count = 0



