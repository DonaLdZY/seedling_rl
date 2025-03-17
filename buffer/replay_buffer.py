import pickle

import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, startup: int = None):
        self.buffer = deque(maxlen=capacity)  # 使用 deque 提高性能
        self.capacity = capacity
        self.startup = min(startup, capacity) if startup is not None else capacity
        self.current_observation = {}
        self.current_reward = {}

    def __len__(self):
        return len(self.buffer)

    def len(self):
        return self.__len__()

    def __sizeof__(self):
        return self.capacity

    def size(self):
        return self.__len__()

    def ready(self):
        return self.size() >= self.startup

    def store(self, observations, actions, action_probs):
        env_ids = observations['env_id']
        rewards = observations['reward']
        next_obs = observations['observation']
        terminated = observations['terminated']
        truncated = observations['truncated']
        dones = np.logical_or(terminated, truncated)  # 向量化 done 计算

        # 已经有过记录的env(不是第一步)
        existing_mask = np.isin(env_ids, list(self.current_observation.keys()))

        for env_id, reward, existing in zip(env_ids, rewards, existing_mask):
            if existing:
                self.current_reward[env_id] += reward
            else:
                self.current_reward[env_id] = 0

        # 只存储已有的 env_id
        if np.any(existing_mask):
            existing_env_ids = env_ids[existing_mask]
            batch = [
                (
                    self.current_observation[env_id][0],  # 之前的 observation
                    self.current_observation[env_id][1],  # 之前的 action
                    rewards[i],
                    next_obs[i],  # next_observation
                    terminated[i],
                    truncated[i],
                )
                for i, env_id in enumerate(existing_env_ids)
            ]
            self.buffer.extend(batch)

        # 批量移除 done 的 env_id
        done_env_ids = env_ids[dones]
        for env_id in done_env_ids:
            self.current_observation.pop(env_id, None)
            score = self.current_reward.pop(env_id, None)
            if score is not None:
                print(f"env{env_id} score: {score}")

        # 批量更新 current_observation（排除已终止的环境）
        valid_mask = ~dones
        self.current_observation.update({
            env_id: (obs, action)
            for env_id, obs, action in zip(env_ids[valid_mask], next_obs[valid_mask], actions[valid_mask])
        })


    def sample(self, n):
        index = np.random.choice(len(self.buffer), n, replace=False)
        batch = [self.buffer[i] for i in index]
        observations, actions, rewards, next_observations, terminated, truncated= zip(*batch)

        return {
            "observation": np.array(observations),
            "action": np.array(actions),
            "reward": np.array(rewards),
            "next_observation": np.array(next_observations),
            "terminated": np.array(terminated),
            "truncated": np.array(truncated),
        }

    def clean(self):
        self.buffer.clear()