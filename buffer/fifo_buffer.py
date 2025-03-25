import time

import numpy as np
from collections import deque
import random

class FIFOBuffer:
    def __init__(self, capacity, startup: int = None, target_fn = None):
        self._last_report_time = time.time()
        self._insertion_counter = 0
        self.capacity = capacity
        self.startup = startup if startup is not None else capacity
        self.buffer = deque(maxlen=capacity)
        self.target_fn = target_fn

    def store(self, data):
        self.logging(len(data))
        self.buffer.extend(data)

    def sample(self, batch_size):
        index = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)

    def ready(self):
        return len(self.buffer) >= self.startup

    def logging(self, count):
        self._insertion_counter += count
        now = time.time()
        if now - self._last_report_time >= 5:
            speed = self._insertion_counter / (now - self._last_report_time)
            print(f"log | buffer | 存入速度: {speed:.2f} items/s")
            self._insertion_counter = 0
            self._last_report_time = now
