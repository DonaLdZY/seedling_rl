import grpc
from concurrent import futures
import threading
import pickle
import time
from queue import Queue

from comm.service_pb2_grpc import LearnerServicer, add_LearnerServicer_to_server
from comm.service_pb2 import Request, Response
from learner.incomplete_trajectories_buffer import IncompleteTrajectoriesBuffer as Incomplete

def run_learner(model, buffer, port="50051"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_LearnerServicer_to_server(Learner(model=model, buffer=buffer), server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    print('gRPC 服务端已开启，端口为' + str(port) + '...')
    server.wait_for_termination()

class Learner(LearnerServicer):
    def __init__(self, model, buffer):
        self._last_report_time = time.time()
        self._insertion_counter = 0

        self.model = model
        self.complete_queue = Queue()
        self.incomplete = Incomplete(output_queue=self.complete_queue)
        self.buffer = buffer

        self.training_thread = threading.Thread(target=self._training_thread, daemon=True)
        self.training_thread.start()

        self.data_prefetching_thread = threading.Thread(target=self._data_prefetching_thread, daemon=True)
        self.data_prefetching_thread.start()

    def get_action(self, observation):
        action, action_prob = self.model.get_action(observation[1])
        self.incomplete.store(observation, action, action_prob)

        self._insertion_counter += observation[0].shape[0]
        now = time.time()
        if now - self._last_report_time >= 1:
            speed = self._insertion_counter / (now - self._last_report_time)
            print(f"吞吐速度: {speed:.2f} items/s")
            self._insertion_counter = 0
            self._last_report_time = now

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
                if train_step % 1000 == 0:
                    print(">>> train: ", train_step, " loss: ",loss.item())
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



