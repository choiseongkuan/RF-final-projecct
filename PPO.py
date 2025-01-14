import torch
import torch.nn.functional as F
import Agent
import Buffer
from torch.distributions import Normal
import numpy as np


class PPO:
    """
    实现Clipped PPO算法所需函数
    """

    def __init__(self, state_dim, action_dim, adam_stepsize, horizon, tricks):
        """
        初始化
        :param state_dim: 环境状态
        :param action_dim: 动作
        :param adam_stepsize: 超参数 adam_stepsize
        :param horizon: 超参数 horizon
        :param tricks: 是否启动trick
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")  # 使用 GPU
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")  # 使用 CPU
            print("Using CPU")

        self.tricks = tricks
        self.agent = Agent.Agent(state_dim, action_dim, adam_stepsize, device, tricks)
        self.buffer = Buffer.Buffer(horizon, state_dim, action_dim, device)
        self.device = device
        self.horizon = int(horizon)
        self.tricks = tricks

    def select_action(self, state):
        """
        根据当前策略选择动作。

        参数：
        - state : 环境状态。

        返回：
        - action: 采样动作。
        - action_log_pi: 动作的 log 概率。
        """
        state = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1).to(self.device)
        mean, std = self.agent.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action_log_pi = dist.log_prob(action)
        return action.detach().cpu().numpy().squeeze(0), action_log_pi.detach().cpu().numpy().squeeze(0)

    def evaluate_action(self, state):
        """
        评估当前策略下的动作。

        参数：
        - state: 环境状态。

        返回：
        - action: 策略输出的动作均值。
        """
        state = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1).to(self.device)
        mean, _ = self.agent.actor(state)
        return mean.detach().cpu().numpy().squeeze(0)

    def add(self, state, action, reward, next_state, done, action_log_pi, adv_dones):
        """
        采样的经验加入缓冲区
        """
        self.buffer.add(state, action, reward, next_state, done, action_log_pi, adv_dones)

    def update(self, minibatch_size, gamma, GAE_parameter, clip_param, Num_epochs, entropy_coefficient):
        """
        策略优化。

        参数：
        - minibatch_size: 超参数minibatch_size
        - gamma: 超参数gamma折扣因子。
        - GAE_parameter: 超参数GAE_parameter
        - clip_param: 超参数Clipped PPO 截断量。
        - Num_epochs: 超参数N. epoch每次优化的 epoch 数量
        - entropy_coefficient (float): 超参数entropy_coefficient熵项系数
        """
        state, action, reward, next_state, done, action_log_pi, adv_dones = self.buffer.all()
        # 计算 GAE 和目标值
        with torch.no_grad():
            adv = np.zeros(self.horizon)
            gae = 0
            vs = self.agent.critic(state)
            vs_ = self.agent.critic(next_state)
            td_delta = reward + gamma * (1.0 - done) * vs_ - vs
            td_delta = td_delta.reshape(-1).cpu().detach().numpy()
            adv_dones = adv_dones.reshape(-1).cpu().detach().numpy()
            for i in reversed(range(self.horizon)):
                gae = td_delta[i] + gamma * GAE_parameter * gae * (1.0 - adv_dones[i])
                adv[i] = gae
            adv = torch.as_tensor(adv, dtype=torch.float32).reshape(-1, 1).to(self.device)
            v_target = adv + vs
            if self.tricks["advantage_normalization"]:  # 论文中提到此方法有利于PG的训练
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8))

        # 策略优化
        for _ in range(Num_epochs):
            # 随机打乱样本生成batch
            shuffled_indices = np.random.permutation(self.horizon)
            indexes = [shuffled_indices[i:i + minibatch_size] for i in range(0, self.horizon, minibatch_size)]
            # 注意这里的实现的先更新actor在更新critic
            for index in indexes:
                # 更新actor
                mean, std = self.agent.actor(state[index])
                dist_now = Normal(mean, std)
                dist_entropy = dist_now.entropy().sum(dim=1,
                                                      keepdim=True)
                action_log_pi_now = dist_now.log_prob(action[index])

                # ratio = pi_now/pi_old = exp(log(a)-log(b))
                ratios = torch.exp(action_log_pi_now.sum(dim=1, keepdim=True) - action_log_pi[index].sum(dim=1,
                                                                                                         keepdim=True))
                policy_gain_unclipped = ratios * adv[index]
                policy_gain_clipped = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * adv[index]
                actor_loss = -torch.min(policy_gain_unclipped,
                                        policy_gain_clipped).mean() - entropy_coefficient * dist_entropy.mean()
                self.agent.update_actor(actor_loss)
                # 更新critic
                v_s = self.agent.critic(state[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                self.agent.update_critic(critic_loss)

        # 清空buffer重新采样
        self.buffer.clear()
