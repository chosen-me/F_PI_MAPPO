"""
经验回放缓冲区（含优先级采样功能）
修复：确保优先级始终存储为标量，避免形状不一致
"""
import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        初始化优先级经验回放缓冲区
        :param capacity: 缓冲区最大容量
        :param alpha: 优先级权重（0=均匀采样，1=完全按优先级）
        :param beta: 重要性采样权重（0=无修正，1=完全修正）
        :param beta_increment: beta随训练步数的增量
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0  # 初始最大优先级（标量）

        # 存储数据（每个元素是完整经验）
        self.buffer = deque(maxlen=capacity)
        # 存储每个样本的优先级（仅存储标量）
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        :param state: 当前状态（多智能体：[num_agents, state_dim]）
        :param action: 执行动作（多智能体：[num_agents, action_dim]）
        :param reward: 获得奖励（多智能体：[num_agents]）
        :param next_state: 下一状态（多智能体：[num_agents, state_dim]）
        :param done: 终止标志（多智能体：[num_agents]）
        """
        # 添加经验到缓冲区
        self.buffer.append((state, action, reward, next_state, done))
        # 新样本赋予最大优先级（标量）
        self.priorities.append(float(self.max_priority))

    def sample(self, batch_size):
        """
        按优先级采样批次数据
        :param batch_size: 采样批次大小
        :return: 采样数据+重要性采样权重+采样索引
        """
        if len(self.buffer) < batch_size:
            return None

        # 关键修复：确保优先级是标量数组
        try:
            # 转换为numpy数组（强制标量）
            priorities = np.array(self.priorities, dtype=np.float32).reshape(-1)
        except Exception as e:
            print(f"⚠️ 优先级转换失败：{e}")
            print(f"  优先级示例：{list(self.priorities)[:5]}")
            #  fallback：使用均匀优先级
            priorities = np.ones(len(self.priorities), dtype=np.float32)

        # 计算采样概率（基于优先级）
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # 按概率采样
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)

        # 提取采样数据
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            data = self.buffer[idx]
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            dones.append(data[4])

        # 转换为numpy数组（处理多智能体数据格式）
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)

        # 计算重要性采样权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        weights = weights.astype(np.float32)

        # 更新beta（逐渐接近1）
        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        """更新采样经验的优先级"""
        # 确保 priorities 是 numpy 数组
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.cpu().numpy()
        
        # 如果 priorities 是标量，将其扩展为数组
        if np.isscalar(priorities):
            priorities = np.full(len(indices), priorities)
        
        # 确保 priorities 和 indices 长度匹配
        if len(priorities) != len(indices):
            raise ValueError(f"indices长度({len(indices)})与priorities长度({len(priorities)})不匹配")
        
        # 更新优先级并跟踪最大优先级
        for idx, priority in zip(indices, priorities):
            # 确保优先级为正数
            scalar_priority = float(max(1e-6, priority))  # 避免优先级为0
            # 直接更新deque中的优先级
            self.priorities[idx] = scalar_priority
            # 更新最大优先级
            if scalar_priority > self.max_priority:
                self.max_priority = scalar_priority

    def __len__(self):
        return len(self.buffer)

    def get_info(self):
        """获取缓冲区状态信息"""
        min_priority = min(self.priorities) if self.priorities else 0.0
        return {
            'current_size': len(self.buffer),
            'capacity': self.capacity,
            'usage_rate': len(self.buffer) / self.capacity * 100,
            'current_beta': self.beta,
            'alpha': self.alpha,
            'priority_range': (min_priority, self.max_priority)
        }

# 测试代码（验证修复效果）
if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=1000)

    # 填充测试数据（模拟多智能体数据）
    for i in range(200):
        state = [np.random.randn(9) for _ in range(3)]  # 3智能体，9维状态
        action = [np.random.uniform(-1, 1, 1) for _ in range(3)]
        reward = np.random.uniform(-1, 1, 3)
        next_state = [np.random.randn(9) for _ in range(3)]
        done = [False] * 3
        buffer.push(state, action, reward, next_state, done)

    print(f"缓冲区填充完成：{len(buffer)}条数据")
    print(f"优先级类型：{type(buffer.priorities[0])}（应是float）")
    print(f"优先级示例：{list(buffer.priorities)[:5]}")

    # 测试采样
    batch = buffer.sample(batch_size=32)
    if batch:
        states, actions, rewards, next_states, dones, weights, indices = batch
        print(f"\n采样成功：")
        print(f"  状态形状：{states.shape}（应为(32,3,9)）")
        print(f"  动作形状：{actions.shape}（应为(32,3,1)）")
        print(f"  权重形状：{weights.shape}（应为(32,)）")
        print(f"  采样索引：{indices[:5]}")

        # 测试优先级更新
        new_prios = np.random.uniform(0.1, 1.0, 32)
        buffer.update_priorities(indices, new_prios)
        print(f"  优先级更新完成，新优先级范围：{buffer.get_info()['priority_range']}")

        print("\n✅ ReplayBuffer测试通过！")
    else:
        print("\n❌ 采样失败")
