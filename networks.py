"""
Actor-Critic神经网络架构（符合CTDE框架要求）
- Actor：分散执行，每个智能体独立网络
- Critic：集中训练，输入所有智能体的状态和动作拼接
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor网络（策略网络）
    输入：单个智能体的9维状态
    输出：1维连续动作（归一化到[-1,1]）
    """

    def __init__(self, state_dim=9, action_dim=1, hidden_dim=64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出限制在[-1,1]
        )

        # 初始化权重（提升训练稳定性）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, state):
        """前向传播"""
        return self.network(state)


class Critic(nn.Module):
    """
    Critic网络（价值网络）
    输入：所有智能体的状态拼接 + 所有智能体的动作拼接
    输出：联合动作的Q值
    """

    def __init__(self, total_state_dim=27, total_action_dim=3, hidden_dim=256):
        super(Critic, self).__init__()
        input_dim = total_state_dim + total_action_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出Q值
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, states, actions):
        """
        前向传播
        :param states: 所有智能体的状态拼接 [batch_size, total_state_dim]
        :param actions: 所有智能体的动作拼接 [batch_size, total_action_dim]
        :return: Q值 [batch_size, 1]
        """
        x = torch.cat([states, actions], dim=1)
        return self.network(x)


# 测试代码（验证网络输出）
if __name__ == "__main__":
    # 初始化网络
    actor = Actor(state_dim=9, action_dim=1)
    critic = Critic(total_state_dim=9 * 3, total_action_dim=1 * 3)

    # 测试Actor网络
    test_state = torch.randn(1, 9)  # 单个智能体状态
    action = actor(test_state)
    print("Actor网络测试：")
    print(f"  输入形状：{test_state.shape}")
    print(f"  输出形状：{action.shape}")
    print(f"  输出范围：[{action.min().item():.3f}, {action.max().item():.3f}]（期望[-1,1]）")

    # 测试Critic网络
    test_states = torch.randn(1, 27)  # 3个智能体状态拼接
    test_actions = torch.randn(1, 3)  # 3个智能体动作拼接
    q_value = critic(test_states, test_actions)
    print("\nCritic网络测试：")
    print(f"  状态输入形状：{test_states.shape}")
    print(f"  动作输入形状：{test_actions.shape}")
    print(f"  Q值输出：{q_value.item():.3f}")

    # 测试参数数量
    print(f"\n网络参数统计：")
    print(f"  Actor参数数：{sum(p.numel() for p in actor.parameters()):,}")
    print(f"  Critic参数数：{sum(p.numel() for p in critic.parameters()):,}")