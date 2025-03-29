import numpy as np
import random

import torch

from utils.sample_to_tensor import sample_to_tensor
class PrioritizedReplayBuffer:
    def __init__(self, capacity, max_usage=64, startup=None, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.capacity = capacity
        self.startup = startup if startup is not None else capacity
        self.max_usage = max_usage  # 每个样本最大使用次数，-1 表示无限制
        self.alpha = alpha  # 用于调整优先级的权重，0表示普通随机采样，1表示完全按照优先级采样
        self.beta = beta  # 重要性采样修正权重
        self.beta_increment = beta_increment  # beta值随时间逐渐增加
        self.epsilon = epsilon  # 避免优先级为0

        self.buffer = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.ids = np.arange(capacity)  # 追踪样本唯一ID
        self.usage = np.zeros(capacity, dtype=int)  # 记录样本被采样的次数
        self.sample_ids = None  # 记录最近一次采样的ID
        self.valid_indices = set()   # 维护可用样本索引

        self.position = 0

    def add(self, sample, priority=1.0):
        max_priority = self.priorities.max() if len(self.valid_indices)>0 else priority

        if self.buffer[self.position] is not None:
            self.valid_indices.discard(self.position)  # 旧样本被覆盖，移出索引

        self.buffer[self.position] = sample
        self.priorities[self.position] = max_priority  # 新样本赋予最大优先级
        self.ids[self.position] = self.position  # 记录样本唯一ID
        self.usage[self.position] = 0  # 重置样本使用计数
        self.valid_indices.add(self.position)  # 新样本加入索引

        self.position = (self.position + 1) % self.capacity

    def store(self, samples):
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size):
        if len(self.valid_indices) < batch_size:
            raise ValueError("Not enough valid samples")

        valid_indices = np.array(list(self.valid_indices))
        priorities = self.priorities[valid_indices] ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(valid_indices, batch_size, p=probabilities)

        self.sample_ids = self.ids[indices]  # 记录样本ID

        # 更新样本使用次数，并移除超限样本
        for idx in indices:
            self.usage[idx] += 1
            if self.max_usage > 0 and self.usage[idx] >= self.max_usage:
                self.valid_indices.discard(idx)  # 移除超限样本

        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.valid_indices) * probabilities) ** (-self.beta)
        weights /= weights.max()  # 归一化

        self.beta = min(1.0, self.beta + self.beta_increment)  # 逐渐增加beta

        return zip(*samples), torch.from_numpy(weights)

    def update_priorities(self, td_errors):
        if self.sample_ids is None:
            return

        for sid, td_error in zip(self.sample_ids, td_errors):
            self.priorities[sid] = np.abs(td_error) + self.epsilon

        self.sample_ids = None  # 清除采样ID，防止误用

    def ready(self):
        return len(self.valid_indices) >= self.startup
    def full(self):
        return len(self.valid_indices) >= self.capacity