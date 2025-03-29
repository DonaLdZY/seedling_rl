import pickle
import numpy as np
import redis

class RedisBufferManager:
    def __init__(self, host='localhost', port=6379, db=0, max_memory=1024):
        """
        轨迹专用缓冲管理器
        :param max_memory: 最大内存限制（单位MB）
        """
        self.client = redis.Redis(host=host, port=port, db=db)
        self.buffer_key = "trajectory_buffer"
        self.max_memory = max_memory * 1024 * 1024  # 转换为字节
        self._init_capacity()

    def _init_capacity(self):
        """基于示例轨迹初始化容量估算"""
        sample_traj = {
            'state': [np.zeros((84,84), dtype=np.float32)],
            'action': [np.array([0], dtype=np.int32)],
            'reward': [np.array([0.0], dtype=np.float32)]
        }
        sample_size = len(self._serialize(sample_traj))
        self.traj_capacity = self.max_memory // sample_size

    def _serialize(self, trajectory):
        """专用轨迹序列化方法"""
        serialized = {}
        for key, array_list in trajectory.items():
            # 每个列表元素独立序列化
            serialized[key] = [
                {
                    'data': arr.tobytes(),
                    'dtype': arr.dtype.str,
                    'shape': arr.shape
                }
                for arr in array_list
            ]
        return pickle.dumps(serialized, protocol=5)

    def _deserialize(self, data_bytes):
        """专用轨迹反序列化方法"""
        serialized = pickle.loads(data_bytes)
        trajectory = {}
        for key, array_list in serialized.items():
            trajectory[key] = [
                np.frombuffer(item['data'],
                dtype=np.dtype(item['dtype'])).reshape(item['shape'])
                for item in array_list
            ]
        return trajectory

    def store_trajectory(self, trajectory):
        # 内存控制：先进先出策略
        current_count = self.client.llen(self.buffer_key)
        if current_count >= self.traj_capacity:
            self.client.ltrim(self.buffer_key, 1, -1)  # 移除最旧轨迹

        # 原子化存储
        pipe = self.client.pipeline()
        pipe.rpush(self.buffer_key, self._serialize(trajectory))
        pipe.execute()

    def retrieve_trajectories(self, batch_size=1):
        """获取指定数量的完整轨迹"""
        with self.client.pipeline() as pipe:
            pipe.lrange(self.buffer_key, 0, batch_size-1)
            pipe.ltrim(self.buffer_key, batch_size, -1)
            items, _ = pipe.execute()
        return [self._deserialize(x) for x in items]

    def current_capacity(self):
        return {
            'max_trajectories': self.traj_capacity,
            'current_count': self.client.llen(self.buffer_key)
        }