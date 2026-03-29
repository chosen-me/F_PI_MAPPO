"""
Ornstein-Uhlenbeck噪声实现（完整标准版本）
用于强化学习连续动作空间探索，每个智能体独立实例
参数：theta=0.15, sigma=0.2, dt=0.01（符合项目要求）
"""
import numpy as np


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=0.01):
        """
        初始化OU噪声
        :param action_dim: 动作维度
        :param mu: 长期均值
        :param theta: 回复速度参数
        :param sigma: 波动率参数
        :param dt: 时间步长
        """
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.copy(self.mu)  # 初始状态

    def reset(self):
        """每个episode开始时重置噪声状态到均值"""
        self.state = np.copy(self.mu)

    def sample(self):
        """生成OU噪声样本（遵循标准公式）"""
        # OU过程公式：dx = theta*(mu - x)*dt + sigma*sqrt(dt)*N(0,1)
        dx = self.theta * (self.mu - self.state) * self.dt
        random_part = self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state += dx + random_part
        return self.state

    def scaled_sample(self, scale=1.0):
        """生成带缩放因子的噪声（用于训练衰减）"""
        return self.sample() * scale


# 测试代码（验证噪声特性）
if __name__ == "__main__":
    # 测试1维噪声（加速度控制）
    ou_noise = OUNoise(action_dim=1)
    samples = [ou_noise.sample()[0] for _ in range(1000)]

    print("OU噪声测试结果：")
    print(f"均值：{np.mean(samples):.4f}（期望接近0）")
    print(f"标准差：{np.std(samples):.4f}（期望~0.2）")
    print(f"相邻样本相关性：{np.corrcoef(samples[:-1], samples[1:])[0, 1]:.4f}（期望>0）")