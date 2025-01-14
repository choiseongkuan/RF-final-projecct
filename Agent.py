import model
import torch
from torch.nn.utils import clip_grad_norm_


class Agent:
    """
    智能体
    """

    def __init__(self, state_dim, action_dim, adam_stepsize, device, tricks):
        """
        初始化智能体

        参数：
        - state_dim: 观察空间的维度。
        - action_dim: 动作空间的维度。
        - adam_stepsize: Adam 优化器的学习率。
        - device: 计算设备（CPU 或 GPU）。
        - tricks: 是否启用某些trick
        """
        # 初始化 actor 网络（策略网络）并移动到指定设备
        self.actor = model.Actor(state_dim, action_dim, tricks).to(device)
        # 初始化 critic 网络（值函数网络）并移动到指定设备
        self.critic = model.Critic(state_dim, tricks).to(device)
        # 为 actor 网络创建 Adam 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=adam_stepsize)
        # 为 critic 网络创建 Adam 优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=adam_stepsize)

    def update_actor(self, loss):
        """
        更新actor的参数。

        参数：
        - loss: 计算得到的损失值。
        """
        # 清空上一次计算的梯度
        self.actor_optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 对梯度进行裁剪以防止梯度爆炸
        clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        # 更新网络参数
        self.actor_optimizer.step()

    def update_critic(self, loss):
        """
        更新 critic 网络的参数。

        参数：
        - loss (torch.Tensor): 计算得到的损失值。
        """
        # 清空上一次计算的梯度
        self.critic_optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 对梯度进行裁剪以防止梯度爆炸
        clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        # 更新网络参数
        self.critic_optimizer.step()
