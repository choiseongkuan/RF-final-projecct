import gymnasium as gym
import numpy as np
import torch
import PPO
import pickle


class MovingMeanStd:
    """
    计算数据的动态均值和标准差
    """

    def __init__(self, shape):
        """
        初始化动态均值和标准差

        参数：
        - shape: 输入数据的维度
        """
        self.n = 0  # 样本计数
        self.mean = np.zeros(shape)  # 动态均值初始化为零
        self.S = np.zeros(shape)  # 动态方差累计值初始化为零
        self.std = np.sqrt(self.S)  # 动态标准差，初始化为零

    def update(self, x):
        """
        更新动态均值和标准差

        参数：
        - x: 新增的数据样本
        """
        x = np.array(x)  # 确保输入是 numpy 数组
        self.n += 1  # 样本计数加 1

        if self.n == 1:  # 如果是第一个样本
            self.mean = x  # 均值直接设为当前样本值
            self.std = x  # 标准差设为当前样本值
        else:
            old_mean = self.mean.copy()  # 保存旧的均值
            # 更新均值
            self.mean = old_mean + (x - old_mean) / self.n
            # 更新累积平方差
            self.S = self.S + (x - old_mean) * (x - self.mean)
            # 更新标准差
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    """
    用于对输入数据进行归一化处理
    """

    def __init__(self, shape):
        """
        初始化归一化类

        参数：
        - shape (tuple): 输入数据的维度
        """
        self.running_ms = MovingMeanStd(shape=shape)  # 使用 MovingMeanStd 来动态更新均值和标准差

    def __call__(self, x, update=True):
        """
        对输入数据进行归一化

        参数：
        - x: 输入数据
        - update: 是否更新动态均值和标准差

        返回：
        - x: 归一化后的数据
        """
        if update:  # 如果需要更新动态统计量
            self.running_ms.update(x)

        # 对数据进行归一化，添加小值以避免除零
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x


def get_env(env_name):
    """
    创建指定名称的环境，并提取其关键信息

    参数：
    - env_name: 环境名称

    返回：
    - env: 创建的 Gym 环境实例
    - dim_info: 包含观测空间维度和动作空间维度的列表
    - max_action: 连续动作空间的最大值
    """
    # 创建环境实例
    env = gym.make(env_name)
    # 获取观测空间的维度（假设是连续的）
    state_dim = env.observation_space.shape[0]
    # 获取动作空间的维度（假设是连续的）
    action_dim = env.action_space.shape[0]
    # 将维度信息存储为列表
    dim_info = [state_dim, action_dim]
    # 获取动作空间的最大值（假设是对称的连续动作空间）
    max_action = env.action_space.high[0]
    # 返回环境实例、维度信息和动作空间的最大值
    return env, dim_info, max_action


def setup_environment(args):
    # 配置训练环境，初始化相关设置
    env, dim_info, max_action = get_env(args["env_name"])
    # 设置随机数种子，确保实验可复现
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    # 确保深度学习结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return env, dim_info, max_action


def initialize_policy(dim_info, args, tricks):
    # 初始化 PPO 算法
    return PPO.PPO(
        dim_info[0], dim_info[1], args["Adam_stepsize"],
        args["Horizon"], tricks
    )


def main_training_loop(env, dim_info, max_action, PPO_policy, args, tricks):
    # 主训练循环
    step = 0
    episode_num = 0
    episode_reward = 0
    state, info = env.reset(seed=args["seed"])

    # 状态和奖励归一化模块（可选）
    state_norm = Normalization(shape=dim_info[0]) if tricks['state_normalization'] else None
    reward_norm = Normalization(shape=1) if tricks['reward_normalization'] else None

    max_step = args["train_timestep"]
    x, y = [], []  # 用于记录训练过程中的奖励

    while step <= max_step:
        step += 1

        # 策略选择动作
        action, action_log_pi = PPO_policy.select_action(state)
        action_ = np.clip(action * max_action, -max_action, max_action)

        # 与环境交互
        next_state, reward, terminated, truncated, infos = env.step(action_)

        # 归一化状态和奖励（如果启用）
        if state_norm:
            next_state = state_norm(next_state)
        if reward_norm:
            reward = reward_norm(reward)

        done = terminated or truncated
        done_bool = terminated  # truncated 表示达到最大步数

        # 保存经验
        PPO_policy.add(state, action, reward, next_state, done_bool, action_log_pi, done)

        episode_reward += reward
        state = next_state

        # 当前 episode 结束时的处理
        if done:
            if (episode_num + 1) % 100 == 0:
                print(f"Episode: {episode_num + 1}, Reward: {episode_reward}")

            x.append(step)
            y.append(episode_reward)

            episode_num += 1
            state, info = env.reset(seed=args["seed"])
            episode_reward = 0

        # 达到更新步数时更新策略
        if step % args["Horizon"] == 0:
            print(f"Step: {step}")
            PPO_policy.update(
                args["Minibatch_size"], args["Discount"], args["GAE_parameter"],
                args["clip_parameter"], args["Num_epochs"], args["entropy_coefficient"]
            )

    # 保存训练结果
    with open('data2.pkl', 'wb') as file:
        pickle.dump((x, y), file)


args = {
    "env_name": "Hopper-v5",
    "Horizon": 2048,
    "Adam_stepsize": 3e-4,
    "Num_epochs": 10,
    "Minibatch_size": 64,
    "Discount": 0.99,
    "GAE_parameter": 0.95,
    "seed": 100,
    "train_timestep": 10000,
    "clip_parameter": 0.2,
    "entropy_coefficient": 0.01,
    "advantage_normalization": True,
    "state_normalization": True,
    "reward_normalization": True,
    "tanh_activation_function": True
}
trick_item = ["advantage_normalization", "state_normalization", "reward_normalization", "tanh_activation_function"]
tricks = dict((key, args[key]) for key in trick_item)

if __name__ == "__main__":
    env, dim_info, max_action = setup_environment(args)
    PPO_policy = initialize_policy(dim_info, args, tricks)
    main_training_loop(env, dim_info, max_action, PPO_policy, args, tricks)
