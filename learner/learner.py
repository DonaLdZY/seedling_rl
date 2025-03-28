import grpc
from comm.service_pb2_grpc import LearnerServicer, add_LearnerServicer_to_server
from comm.service_pb2 import Response
import pickle

import threading
from concurrent import futures
from learner.state_buffer import StateBuffer
from queue import Queue, Empty
import time

import numpy as np
import torch

from utils.logger import SpeedLogger, CycleInfoLogger, IncreasingSpeedLogger

class Learner(LearnerServicer):
    def __init__(self, model, buffer,
                 trajectory_process,
                 pred_batching:bool = True,
                 **kwargs):
        # learner structure
        self.model = model
        self.complete_queue = Queue()
        self.state_buffer = StateBuffer(output_queue=self.complete_queue)
        self.buffer = buffer
        self.trajectory_process = trajectory_process
        self.train_batch_size = kwargs.get('train_batch_size', 64)

        # batching layer
        self.pred_batching = pred_batching
        if self.pred_batching:
            self.pred_queue = Queue()
            self.pred_batch_size = kwargs.get('pred_batch_size', 32)
            self.pred_timeout = kwargs.get('pred_timeout', 0.1)
            self.pred_batch_size_alpha = kwargs.get('pred_batch_size_alpha', 0.1)
            self.pred_expect_batch_size = self.pred_batch_size
            self.pred_last_batch_size = [1]
            self.batch_thread = threading.Thread(target=self._batching_thread, daemon=True)

        # training/data preparing threads
        self.training_thread = threading.Thread(target=self._training_thread, daemon=True)
        self.data_preparing_thread = threading.Thread(target=self._data_preparing_thread, daemon=True)


        # log
        self.pred_throughput_logger = SpeedLogger("predicting speed |", "actions/s")
        self.train_throughput_logger = IncreasingSpeedLogger("training speed |", "steps/s")
        self.batch_state_logger = CycleInfoLogger()

    def get_action(self, observation):
        if self.pred_batching:
            future = futures.Future()
            self.pred_queue.put((observation, future))
            return future.result()
        else:
            return self.predict(observation)

    def predict(self, observation):
        action, info = self.model.get_action(observation[1])
        self.state_buffer.store(observation, action, info)
        self.pred_throughput_logger.log(observation[0].shape[0])
        return action

    def _batching_thread(self):
        print("batching thread start...")
        while True:
            tasks = []
            batch_sizes = []
            batch_size_sum = 0
            start_time = time.time()
            remaining = self.pred_timeout - (time.time() - start_time)
            while batch_size_sum < self.pred_batch_size and remaining > 0:
                try:
                    # 非阻塞获取队列中的任务，直到队列为空
                    while batch_size_sum < self.pred_batch_size:
                        observation, future = self.pred_queue.get_nowait()
                        tasks.append((observation, future))
                        batch_sizes.append(observation[1].shape[0])
                        batch_size_sum += observation[1].shape[0]
                except Empty:
                    # 队列为空，等待剩余超时时间
                    remaining = self.pred_timeout * (1 - batch_size_sum / self.pred_expect_batch_size) - (time.time() - start_time)
                    if remaining <= 0:
                        break
                    try:
                        observation, future = self.pred_queue.get(timeout=remaining)
                        tasks.append((observation, future))
                        batch_sizes.append(observation[1].shape[0])
                        batch_size_sum += observation[1].shape[0]
                    except Empty:
                        break

            if len(self.pred_last_batch_size) >= 1 and batch_size_sum not in self.pred_last_batch_size:
                torch.cuda.empty_cache()
                self.pred_last_batch_size = [batch_size_sum]
            else:
                if batch_size_sum not in self.pred_last_batch_size:
                    self.pred_last_batch_size.append(batch_size_sum)

            self.pred_expect_batch_size = (min(max(batch_size_sum, 1), self.pred_batch_size) * self.pred_batch_size_alpha
                                           + self.pred_expect_batch_size * (1 - self.pred_batch_size_alpha))

            if not tasks:
                continue

            env_id_list, obs_list, reward_list, terminated_list, truncated_list = [], [], [], [], []
            for observation, _ in tasks:
                env_id_list.append(observation[0])
                obs_list.append(observation[1])
                reward_list.append(observation[2])
                terminated_list.append(observation[3])
                truncated_list.append(observation[4])
            env_ids = np.concatenate(env_id_list, axis=0)
            obs = np.concatenate(obs_list, axis=0)
            rewards = np.concatenate(reward_list, axis=0)
            terminateds = np.concatenate(terminated_list, axis=0)
            truncateds = np.concatenate(truncated_list, axis=0)

            batch_actions = self.predict((
                env_ids,
                obs,
                rewards,
                terminateds,
                truncateds,
            ))

            # 按各请求原始 batch 大小拆分预测结果，并依次处理
            start = 0
            for (obs, future), size in zip(tasks, batch_sizes):
                end = start + size
                actions = batch_actions[start:end]
                future.set_result(actions)
                start = end

            self.batch_state_logger.log(f"expect batch_size | {self.pred_expect_batch_size:.2f}")

    def _data_preparing_thread(self):
        print("data preparing thread start...")
        while True:
            complete_trajectory = self.complete_queue.get()
            data_store = self.trajectory_process(complete_trajectory)
            self.buffer.store(data_store)

    def _training_thread(self):
        print("training thread start...")
        while True:
            if self.buffer.ready():
                batch = self.buffer.sample(self.train_batch_size)
                train_step, loss = self.model.train(batch)
                self.train_throughput_logger.log(train_step)
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

    def run(self, port="50051", max_workers=32):
        self.training_thread.start()
        self.data_preparing_thread.start()
        if self.pred_batching:
            self.batch_thread.start()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        add_LearnerServicer_to_server(self, server)
        server.add_insecure_port('[::]:' + str(port))
        server.start()
        print('gRPC 服务端已开启，端口为' + str(port) + '...')
        server.wait_for_termination()





