import numpy as np
import torch


class Buffer:
    """
    经验回放缓冲区
    定义一些基本的操作如加入缓冲区、晴空缓冲区等
    """

    def __init__(self, capacity, state_dim, act_dim, device):
        self._index = 0
        self._size = 0
        self.device = device
        self.capacity = capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, act_dim))
        self.rewards = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, state_dim))
        self.dones = np.zeros(capacity, dtype=bool)
        self.action_log_probs = np.zeros((capacity, act_dim))
        self.adv_dones = np.zeros(capacity, dtype=bool)

    def add(self, state, action, reward, next_state, done, action_log_probs, adv_done):
        self.state[self._index] = state
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs[self._index] = next_state
        self.dones[self._index] = done
        self.action_log_probs[self._index] = action_log_probs
        self.adv_dones[self._index] = adv_done
        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:  # 循环缓冲区
            self._size += 1

    def clear(self):
        self._index = 0
        self._size = 0

    def all(self):
        state = torch.as_tensor(self.state, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(self.actions, dtype=torch.float32).to(
            self.device)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32).reshape(-1, 1).to(
            self.device)
        next_obs = torch.as_tensor(self.next_obs, dtype=torch.float32).to(
            self.device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32).reshape(-1, 1).to(self.device)
        actions_log_probs = torch.as_tensor(self.action_log_probs, dtype=torch.float32).to(
            self.device)
        adv_dones = torch.as_tensor(self.adv_dones, dtype=torch.float32).reshape(-1, 1).to(self.device)
        return state, actions, rewards, next_obs, dones, actions_log_probs, adv_dones
