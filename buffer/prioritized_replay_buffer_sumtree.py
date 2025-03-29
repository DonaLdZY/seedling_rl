import numpy as np
import random
import torch
from utils.sample_to_tensor import sample_to_tensor


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.depth = int(np.ceil(np.log2(capacity)))
        self.leaf_start = 2 ** self.depth
        self.tree = np.zeros(2 * self.leaf_start - 1, dtype=np.float32)  # 优化存储空间

    def update(self, data_idx, value):
        tree_idx = self.leaf_start + data_idx
        if tree_idx >= len(self.tree):
            return
        delta = value - self.tree[tree_idx]
        self.tree[tree_idx] = value
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get_total(self):
        return self.tree[0]

    def find(self, s):
        idx = 0
        while idx < self.leaf_start - 1:
            left = 2 * idx + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - (self.leaf_start - 1)  # 修正索引偏移


class PrioritizedReplayBuffer:
    def __init__(self, capacity, max_usage=64, startup=None, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.capacity = capacity
        self.startup = startup if startup is not None else capacity
        self.max_usage = max_usage
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # 核心数据结构
        self.buffer = [None] * capacity
        self.sum_tree = SumTree(capacity)
        self.valid_indices = set()  # 有效索引
        self.raw_priorities = np.zeros(capacity, dtype=np.float32)
        self.usage = np.zeros(capacity, dtype=int)
        self.ids = np.arange(capacity)  # 唯一标识符

        # 状态变量
        self.position = 0
        self.valid_size = 0  # 独立维护的有效样本计数器

    def add(self, sample, priority=1.0):
        """添加样本时的精确容量管理"""
        # 处理被覆盖的旧样本
        if self.buffer[self.position] is not None:
            if self.position in self.valid_indices:
                self.valid_indices.remove(self.position)
                self.valid_size -= 1  # 精确递减有效计数
            self.sum_tree.update(self.position, 0)  # 清除旧优先级

        # 添加新样本
        self.buffer[self.position] = sample
        self.raw_priorities[self.position] = priority
        self.sum_tree.update(self.position, (priority + self.epsilon) ** self.alpha)
        self.usage[self.position] = 0

        # 更新有效状态
        self.valid_indices.add(self.position)
        self.valid_size += 1  # 精确递增有效计数

        # 移动指针并处理循环覆盖
        self.position = (self.position + 1) % self.capacity

    def store(self, samples):
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size):
        """带有效性保障的采样方法"""
        if self.valid_size < self.startup:
            raise ValueError(f"Buffer not ready. Valid samples: {self.valid_size}/{self.startup}")

        indices = []
        total_priority = self.sum_tree.get_total()

        # 有效性保障采样循环
        while len(indices) < batch_size:
            s = random.uniform(0, total_priority)
            idx = self.sum_tree.find(s)
            if idx in self.valid_indices:  # 有效性验证
                indices.append(idx)

        # 更新使用计数并处理过期样本
        self.sample_ids = [self.ids[idx] for idx in indices]
        for idx in indices:
            self.usage[idx] += 1
            if self.max_usage > 0 and self.usage[idx] >= self.max_usage:
                self.valid_indices.discard(idx)
                self.valid_size -= 1  # 精确递减
                self.sum_tree.update(idx, 0)  # 使失效样本无法再被采样

        # 重要性采样权重计算
        probabilities = np.array([(self.raw_priorities[idx] + self.epsilon) ** self.alpha / total_priority
                                  for idx in indices])
        weights = (self.valid_size * probabilities) ** (-self.beta)
        weights /= weights.max()  # 归一化

        self.beta = min(1.0, self.beta + self.beta_increment)

        return sample_to_tensor(zip(*[self.buffer[idx] for idx in indices])), torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, td_errors):
        """带有效性检查的优先级更新"""
        valid_updates = 0
        for sid, error in zip(self.sample_ids, td_errors):
            if sid not in self.valid_indices:  # 跳过已失效样本
                continue
            new_p = abs(error) + self.epsilon
            self.raw_priorities[sid] = new_p
            self.sum_tree.update(sid, new_p ** self.alpha)
            valid_updates += 1
        self.sample_ids = None

    def ready(self):
        """精确的有效样本数判断"""
        return self.valid_size >= self.startup
    def full(self):
        return self.valid_size >= self.capacity

    @property
    def real_capacity(self):
        """当前实际有效容量"""
        return min(self.valid_size, self.capacity)

if __name__=="__main__":
    # 测试容量边界条件
    buffer = PrioritizedReplayBuffer(capacity=3, startup=2)
    sample = [1]
    buffer.add(sample)  # valid_size=1
    buffer.add(sample)  # valid_size=2
    assert buffer.ready() == True  # startup=2
    buffer.add(sample)  # valid_size=3
    buffer.add(sample)  # 覆盖位置0，valid_size保持3
    assert buffer.valid_size == 3
    assert buffer.real_capacity == 3