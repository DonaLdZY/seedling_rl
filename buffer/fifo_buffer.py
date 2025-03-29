import numpy as np
import random
from utils.sample_to_tensor import sample_to_tensor
class FIFOBuffer:
    def __init__(self, capacity, max_usage = 64, startup = None):
        self.capacity = capacity
        self.max_usage = max_usage
        self.startup = startup if startup is not None else capacity
        self.buffer = [None] * capacity
        self.usage = np.zeros(capacity, dtype=int)

        self.size = 0  # 当前存储的样本数
        self.next_idx = 0  # 下一个插入的位置
        self.valid_indices = set() # 用一个集合维护还未达到最大使用次数的样本索引

    def add(self, sample):
        if self.buffer[self.next_idx] is not None:
            self.valid_indices.discard(self.next_idx)
        self.buffer[self.next_idx] = sample
        self.usage[self.next_idx] = 0  # 新样本使用次数置0
        self.valid_indices.add(self.next_idx)
        self.size = min(self.size + 1, self.capacity)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def store(self, samples):
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size):
        if len(self.valid_indices) < batch_size:
            raise Exception("valid samples are not enough")
        sampled_indices = random.sample(list(self.valid_indices), batch_size)
        samples = [self.buffer[idx] for idx in sampled_indices]
        if self.max_usage > 0:
            for idx in sampled_indices:
                self.usage[idx] += 1
                # 如果使用次数达到上限，则从有效索引集合中移除
                if self.usage[idx] >= self.max_usage:
                    self.valid_indices.remove(idx)
        return sample_to_tensor(zip(*samples)), None

    def update_priorities(self, td_errors):
        pass

    def ready(self):
        return len(self.valid_indices) >= self.startup
    def full(self):
        return len(self.valid_indices) >= self.capacity

if __name__ == "__main__":
    # test
    rb = FIFOBuffer(capacity=10000, max_usage=5)
    obs = [f"state_{i}" for i in range(5000)]
    actions = [f"action_{i}" for i in range(5000)]
    rewards = [f"reward_{i}" for i in range(5000)]
    next_obs = [f"next_state_{i}" for i in range(5000)]
    dones = [False] * 5000
    samples = list(zip(obs, actions, rewards, next_obs, dones))
    rb.store(samples)
    try:
        batch = rb.sample(32)
        for s in batch:
            print(s)
    except Exception as e:
        print(e)
