import grpc
from comm.service_pb2_grpc import LearnerServicer
from comm.service_pb2 import Response
import pickle

import threading

from learner.incomplete_trajectories_buffer import IncompleteTrajectoriesBuffer as Incomplete
from queue import Queue, Empty
import concurrent.futures
import time

from concurrent import futures
from comm.service_pb2_grpc import add_LearnerServicer_to_server

import numpy as np

class Learner(LearnerServicer):

        def __init__(self, model, buffer,
                     trajectory_process, sample_process,
                     train_batch_size = 64,
                     pred_batch_size=32,
                     pred_wait_timeout=0.005):
            # learner structure
            self.model = model
            self.complete_queue = Queue()
            self.incomplete = Incomplete(output_queue=self.complete_queue)
            self.buffer = buffer
            self.trajectory_process = trajectory_process
            self.sample_process = sample_process
            self.train_batch_size = train_batch_size

            # 新增预测请求 batching 相关属性
            self.pred_queue = Queue()
            self.pred_batch_size = pred_batch_size
            self.pred_wait_timeout = pred_wait_timeout
            self.batch_thread = threading.Thread(target=self._batching_thread, daemon=True)
            self.batch_thread.start()
            # training/data preparing threads
            self.training_thread = threading.Thread(target=self._training_thread, daemon=True)
            self.training_thread.start()
            self.data_preparing_thread = threading.Thread(target=self._data_preparing_thread, daemon=True)
            self.data_preparing_thread.start()

            # log
            self.action_time = time.time()
            self.action_count = 0
            self.train_time = time.time()
            self.train_count = 0
            self.train_loss = []


        def getAction(self, request, context):
            observation = pickle.loads(request.observation)
            action = self.predict(observation)
            _action = pickle.dumps(action)
            return Response(action=_action)
        def getActionStream(self, request_iterator, context):
            for request in request_iterator:
                observation = pickle.loads(request.observation)
                action = self.predict(observation)
                _action = pickle.dumps(action)
                yield Response(action=_action)
        def predict(self, observation):
            future = concurrent.futures.Future()
            self.pred_queue.put((observation, future))
            return future.result()



        def _batching_thread(self):
            """
            预测 batching 线程：
            1. 从队列中收集多个预测请求，每个请求中 obs[1] 已经是一个 batch（形状为 [n, state_dim]）。
            2. 将所有请求的 obs[1] 按 batch 维度拼接，再调用模型的批量预测接口。
            3. 根据每个请求原始的 batch 大小拆分预测结果，并依次调用不完整轨迹存储、日志记录，
               最后将对应结果设置到 Future 中返回。
            """
            # self.pred_queue = Queue()
            # self.pred_batch_size = pred_batch_size
            # self.pred_wait_timeout = pred_wait_timeout
            while True:
                tasks = []
                batch_sizes = []
                start_time = time.time()
                # 等待队列中至少有一个任务到达
                while len(tasks) < self.pred_batch_size:
                    try:
                        observation, future = self.pred_queue.get(timeout=self.pred_wait_timeout)
                        tasks.append((observation, future))
                        batch_sizes.append(observation[1].shape[0])  # 获取当前请求的 batch 大小
                    except Empty:
                        break

                    if time.time() - start_time >= self.pred_wait_timeout:
                        break

                if tasks:
                    # 对批次中的数据进行处理，提取并堆叠各个元素
                    env_ids = np.concatenate([obs[0] for obs, _ in tasks], axis=0)
                    obs = np.concatenate([obs[1] for obs, _ in tasks], axis=0)
                    rewards = np.concatenate([obs[2] for obs, _ in tasks], axis=0)
                    terminateds = np.concatenate([obs[3] for obs, _ in tasks], axis=0).astype(np.float32)
                    truncateds = np.concatenate([obs[4] for obs, _ in tasks], axis=0).astype(np.float32)

                    # 使用模型进行预测
                    batch_actions = self.get_action((
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

        def get_action(self, observation):
            action, info = self.model.get_action(observation[1])
            self.incomplete.store(observation, action, info)
            self.logging_action(observation[0].shape[0])
            return action


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
                    batch = self.sample_process(batch)
                    train_step, loss = self.model.train(batch)
                    self.logging_train(train_step, loss)
                else:
                    time.sleep(10)

        def logging_action(self, action_count):
            self.action_count += action_count
            now=time.time()
            if now - self.action_time >= 5:
                speed = self.action_count / (now - self.action_time)
                print(f"log | action thread | 吞吐速度: {speed:.2f} actions/s")
                self.action_time = now
                self.action_count = 0

        def logging_train(self, train_step, loss):
            self.train_loss.append(loss)
            now = time.time()
            if now - self.train_time >= 5:
                import numpy as np
                speed = (train_step-self.train_count) / (now - self.train_time)
                print(f"log | train thread | 训练速度 : {speed:.2f} steps/s")
                print("log | train thread | 平均loss :", np.mean(np.array(self.train_loss)))
                self.train_time = now
                self.train_count = train_step
                self.train_loss = []




def run_learner(learner, port="50051"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_LearnerServicer_to_server(learner, server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    print('gRPC 服务端已开启，端口为' + str(port) + '...')
    server.wait_for_termination()


