import time

class CycleInfoLogger:
    def __init__(self, cycle = 5.0):
        self.cycle = cycle
        self.count = 0
        self.time = None
    def log(self, info):
        now = time.time()
        if self.time is None:
            self.time = now
        if now - self.time >= self.cycle:
            print(info)
            self.time = now

class AverageLogger:
    def __init__(self, info, unit, cycle = 5.0):
        self.info = info
        self.unit = unit
        self.cycle = cycle
        self.items = []
        self.time = None
    def log(self, item):
        self.items.append(item)
        now = time.time()
        if self.time is None:
            self.time = now
        if now - self.time >= self.cycle:
            average = sum(self.items) / len(self.items)
            info = f"{self.info} {average:.2f} {self.unit}"
            print(info)
            self.items = []
            self.time = now

class SpeedLogger:
    def __init__(self, info, unit, cycle = 5.0):
        self.info = info
        self.unit = unit
        self.cycle = cycle
        self.count = 0
        self.time = None
    def log(self, count=1):
        self.count += count
        now = time.time()
        if self.time is None:
            self.time = now
        if now - self.time >= self.cycle:
            speed = self.count / (now - self.time)
            info = f"{self.info} {speed:.2f} {self.unit}"
            print(info)
            self.time = now
            self.count = 0

class IncreasingSpeedLogger:
    def __init__(self, info, unit, cycle=5.0):
        self.info = info
        self.unit = unit
        self.cycle = cycle
        self.last_count = 0
        self.time = None
    def log(self, count):
        now = time.time()
        if self.time is None:
            self.time = now
        if now - self.time >= self.cycle:
            speed = (count - self.last_count) / (now - self.time)
            info = f"{self.info} {speed:.2f} {self.unit}"
            print(info)
            self.time = now
            self.last_count = count
