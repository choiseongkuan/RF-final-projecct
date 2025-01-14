import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor网络
    在连续环境中，Actor应该用均值和方差来更新以达到好的效果
    """
    def __init__(self, state_dim, action_dim, tricks, hidden_1=128, hidden_2=128):
        """
        使用全连接网络来搭建
        :param state_dim: 环境状态
        :param action_dim: 动作
        :param hidden_1: 全连接层1参数
        :param hidden_2: 全连接层2参数
        """
        super(Actor, self).__init__()
        self.tricks = tricks
        self.fc1 = nn.Linear(state_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.mean = nn.Linear(hidden_2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.activation_function = nn.Tanh() if self.tricks['tanh_activation_function'] else nn.ReLU()  # 有论文提到POO算法比较适合tanh激活函数

    def forward(self, state):
        """
        前向传播定义
        :param state: 环境状态
        :return: 均值和方差
        """
        x = self.activation_function(self.fc1(state))
        x = self.activation_function(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std


class Critic(nn.Module):
    """
    原论文提到，需要引入状态值函数来减小优势函数方差
    """
    def __init__(self, state_dim, tricks, hidden_1=128, hidden_2=128):
        """
        使用全连接网络来搭建
        :param state_dim: 环境状态
        :param hidden_1: 全连接层1参数
        :param hidden_2: 全连接层2参数
        """
        super(Critic, self).__init__()
        self.tricks = tricks
        self.fc1 = nn.Linear(state_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 1)
        self.activation_function = nn.Tanh() if self.tricks['tanh_activation_function'] else nn.ReLU()  # 有论文提到POO算法比较适合tanh激活函数

    def forward(self, state):
        """
        前向传播定义
        :param state: 环境状态
        :return: 状态值函数
        """
        x = self.activation_function(self.fc1(state))
        x = self.activation_function(self.fc2(x))
        value = self.fc3(x)
        return value