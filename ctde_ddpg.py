"""
CTDE-DDPG算法（集中训练-分散执行）
✅ 修复方法缩进问题，确保所有方法属于CTDE_DDPG类
✅ 每个智能体独立OU噪声
✅ 优先级经验回放
✅ 集中式Critic训练，分散式Actor执行
✅ 完整的目标网络软更新
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
import time
from networks import Actor, Critic
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

class CTDE_DDPG:
    def __init__(self, num_agents=3, state_dim=9, action_dim=1, config=None):
        """
        初始化CTDE-DDPG算法
        :param num_agents: 智能体数量（跟随车辆数）
        :param state_dim: 单个智能体状态维度
        :param action_dim: 单个智能体动作维度
        :param config: 算法配置
        """
        # 配置合并
        default_config = self.get_default_config()
        self.config = {**default_config, **(config or {})}

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 算法超参数
        self.gamma = self.config['gamma']  # 折扣因子
        self.tau = self.config['tau']      # 软更新系数
        self.actor_lr = self.config['actor_lr']
        self.critic_lr = self.config['critic_lr']

        # 设备配置（自动检测GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💻 使用计算设备：{self.device}")
        if torch.cuda.is_available():
            print(f"   GPU型号：{torch.cuda.get_device_name(0)}")

        # 初始化网络
        self._init_networks()

        # 初始化OU噪声（每个智能体独立实例）
        self._init_noise()

        # 初始化经验回放缓冲区（优先级）
        self.replay_buffer = ReplayBuffer(
            capacity=self.config['buffer_size'],
            alpha=self.config['prioritized_replay_alpha'],
            beta=self.config['prioritized_replay_beta'],
            beta_increment=self.config['prioritized_replay_beta_increment']
        )

        # 训练状态记录
        self.training_step = 0
        self.episode_count = 0
        self.training_losses = []  # 确保是列表类型

        print(f"✅ CTDE-DDPG算法初始化完成")
        print(f"   智能体数量：{num_agents}，状态维度：{state_dim}，动作维度：{action_dim}")
        print(f"   经验缓冲区容量：{self.config['buffer_size']}")
        print(f"   OU噪声：theta={self.config['ou_theta']}，sigma={self.config['ou_sigma']}")

    def get_default_config(self):
        """算法默认配置"""
        return {
            # 学习率
            'actor_lr': 0.00005,
            'critic_lr': 0.0001,
            # 折扣因子和软更新系数
            'gamma': 0.95,
            'tau': 0.01,
            # 经验回放
            'buffer_size': 100000,
            'batch_size': 512,
            # 优先级回放参数
            'prioritized_replay_alpha': 0.6,
            'prioritized_replay_beta': 0.4,
            'prioritized_replay_beta_increment': 0.001,
            # OU噪声参数
            'ou_theta': 0.15,
            'ou_sigma': 0.2,
            'ou_dt': 0.01,
            'initial_noise_scale': 0.3,
            'noise_decay': 0.99,
            'min_noise_scale': 0.05,
            # 训练参数
            'gradient_clip': 1.0,  # 梯度裁剪阈值
            'target_network_update_freq': 3  # 目标网络更新频率（每步更新）
        }

    def _init_networks(self):
        """初始化Actor和Critic网络"""
        print("📡 初始化神经网络...")

        # Actor网络（每个智能体独立一个）
        self.actors = [Actor(self.state_dim, self.action_dim).to(self.device) for _ in range(self.num_agents)]
        self.actor_targets = [copy.deepcopy(actor).to(self.device) for actor in self.actors]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=self.actor_lr) for actor in self.actors]

        # Critic网络（集中式，输入所有智能体的状态和动作拼接）
        total_state_dim = self.state_dim * self.num_agents
        total_action_dim = self.action_dim * self.num_agents
        self.critic = Critic(total_state_dim, total_action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # 冻结目标网络参数（初始时）
        for target_net in self.actor_targets:
            for param in target_net.parameters():
                param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

        print(f"   Actor网络：{self.num_agents}个独立网络（每个{sum(p.numel() for p in self.actors[0].parameters()):,}参数）")
        print(f"   Critic网络：1个集中式网络（{sum(p.numel() for p in self.critic.parameters()):,}参数）")

    def _init_noise(self):
        """初始化OU噪声（每个智能体独立）"""
        print("🔊 初始化OU噪声...")
        self.ou_noises = [
            OUNoise(
                action_dim=self.action_dim,
                theta=self.config['ou_theta'],
                sigma=self.config['ou_sigma'],
                dt=self.config['ou_dt']
            ) for _ in range(self.num_agents)
        ]

        # 噪声缩放因子（随训练衰减）
        self.noise_scale = self.config['initial_noise_scale']
        self.noise_decay = self.config['noise_decay']
        self.min_noise_scale = self.config['min_noise_scale']

        print(f"   初始噪声缩放：{self.noise_scale}，衰减系数：{self.noise_decay}，最小缩放：{self.min_noise_scale}")

    def get_actions(self, states, exploration=True):
        """
        分散执行：每个智能体基于局部状态选择动作
        :param states: 所有智能体的状态列表 [num_agents, state_dim]
        :param exploration: 是否添加探索噪声
        :return: 所有智能体的动作列表 [num_agents, action_dim]
        """
        actions = []

        for i, (actor, state) in enumerate(zip(self.actors, states)):
            # 转换为张量并移动到设备
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 禁用梯度计算
            with torch.no_grad():
                # Actor网络输出归一化动作（[-1,1]）
                action = actor(state_tensor).cpu().numpy()[0]

            # 添加OU探索噪声
            if exploration and self.noise_scale > 0:
                # 生成OU噪声并缩放
                noise = self.ou_noises[i].sample()
                noisy_action = action + noise * self.noise_scale

                # 裁剪到[-1,1]范围
                action = np.clip(noisy_action, -1.0, 1.0)

            # 记录动作
            actions.append(action)

        return actions

    def reset_noise(self):
        """每个episode开始时重置所有智能体的OU噪声状态"""
        for noise in self.ou_noises:
            noise.reset()
        print(f"🔄 重置OU噪声（当前缩放因子：{self.noise_scale:.3f}）")

    def decay_noise(self):
        """噪声缩放因子随训练衰减（不低于最小值）"""
        self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
        if self.training_step % 100 == 0:
            print(f"📉 噪声缩放因子衰减：{self.noise_scale:.3f}")

    def update(self):
        """
        集中训练：使用全局信息更新网络
        :return: 训练损失字典（含critic_loss、actor_loss等）
        """
        # 检查缓冲区数据是否足够
        if len(self.replay_buffer) < self.config['batch_size']:
            return None

        # 从优先级缓冲区采样
        batch = self.replay_buffer.sample(batch_size=self.config['batch_size'])
        if batch is None:
            return None

        states, actions, rewards, next_states, dones, weights, indices = batch

        # 转换为张量并移动到设备
        states = torch.FloatTensor(states).to(self.device)  # [batch, num_agents, state_dim]
        actions = torch.FloatTensor(actions).to(self.device)  # [batch, num_agents, action_dim]
        rewards = torch.FloatTensor(rewards).to(self.device)  # [batch, num_agents]
        next_states = torch.FloatTensor(next_states).to(self.device)  # [batch, num_agents, state_dim]
        dones = torch.FloatTensor(dones).to(self.device)  # [batch, num_agents]
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)  # [batch, 1]

        batch_size = states.shape[0]

        # ==================== 1. 更新Critic网络（集中式训练）====================
        # 拼接所有智能体的状态和动作（Critic输入格式）
        states_flat = states.view(batch_size, -1)  # [batch, num_agents*state_dim]
        actions_flat = actions.view(batch_size, -1)  # [batch, num_agents*action_dim]
        next_states_flat = next_states.view(batch_size, -1)

        # 计算目标Q值
        with torch.no_grad():
            # 目标Actor生成下一动作（所有智能体）
            next_actions = []
            for i in range(self.num_agents):
                next_action_i = self.actor_targets[i](next_states[:, i])
                next_actions.append(next_action_i)
            next_actions = torch.stack(next_actions, dim=1)  # [batch, num_agents, action_dim]
            next_actions_flat = next_actions.view(batch_size, -1)

            # 目标Critic计算Q值
            target_q = self.critic_target(next_states_flat, next_actions_flat)

            # 总奖励（所有智能体奖励求和）
            total_rewards = rewards.sum(dim=1, keepdim=True)  # [batch, 1]

            # 终止状态处理（只要有一个智能体终止则视为终止）
            done_mask = (1 - dones.max(dim=1, keepdim=True)[0].clamp(max=1.0))

            # 贝尔曼方程：target_q = r + gamma * target_q' * (1-done)
            target_q = total_rewards + self.gamma * target_q * done_mask

        # 计算当前Q值
        current_q = self.critic(states_flat, actions_flat)

        # 计算加权TD误差（用于优先级更新）
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()

        # Critic损失（加权MSE）
        critic_loss = (weights * nn.MSELoss(reduction='none')(current_q, target_q)).mean()

        # 更新Critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['gradient_clip'])
        self.critic_optimizer.step()

        # ==================== 2. 更新Actor网络（分散式执行）====================
        # 冻结Critic参数（仅更新Actor）
        for param in self.critic.parameters():
            param.requires_grad = False

        # 生成当前策略的动作
        new_actions = []
        for i in range(self.num_agents):
            new_action_i = self.actors[i](states[:, i])
            new_actions.append(new_action_i)
        new_actions = torch.stack(new_actions, dim=1)  # [batch, num_agents, action_dim]
        new_actions_flat = new_actions.view(batch_size, -1)

        # Actor损失（最大化Q值，所以取负）
        actor_loss = -self.critic(states_flat, new_actions_flat).mean()

        # 更新所有Actor网络
        for optimizer in self.actor_optimizers:
            optimizer.zero_grad()
        actor_loss.backward()
        # 梯度裁剪
        for actor in self.actors:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), self.config['gradient_clip'])
        for optimizer in self.actor_optimizers:
            optimizer.step()

        # 解冻Critic参数
        for param in self.critic.parameters():
            param.requires_grad = True

        # ==================== 3. 软更新目标网络====================
        if self.training_step % self.config['target_network_update_freq'] == 0:
            self.soft_update(self.critic, self.critic_target)
            for i in range(self.num_agents):
                self.soft_update(self.actors[i], self.actor_targets[i])

        # ==================== 4. 更新缓冲区优先级====================
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)  # +1e-6避免优先级为0

        # ==================== 记录训练状态====================
        self.training_step += 1
        loss_info = {
            'training_step': self.training_step,  # 新增：添加训练步数键
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'avg_q_value': current_q.mean().item(),
            'avg_reward': total_rewards.mean().item(),
            'noise_scale': self.noise_scale,
            'buffer_size': len(self.replay_buffer),
            'beta': self.replay_buffer.beta
        }
        self.training_losses.append(loss_info)

        # 每100步打印训练信息
        if self.training_step % 100 == 0:
            print(f"📊 训练步数{self.training_step}："
                  f"Critic损失={loss_info['critic_loss']:.4f}，"
                  f"Actor损失={loss_info['actor_loss']:.4f}，"
                  f"平均Q值={loss_info['avg_q_value']:.3f}")

        return loss_info

    def soft_update(self, local_model, target_model):
        """
        目标网络软更新
        :param local_model: 本地网络（待更新）
        :param target_model: 目标网络（被更新）
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_models(self, filepath):
        """
        保存模型（含所有Actor、Critic及优化器状态）
        :param filepath: 保存路径
        """
        try:
            # 创建目录（如果不存在）
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # 构建检查点
            checkpoint = {
                'actors': [actor.state_dict() for actor in self.actors],
                'actor_targets': [target.state_dict() for target in self.actor_targets],
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                'noise_scale': self.noise_scale,
                'ou_noise_states': [noise.state for noise in self.ou_noises],
                'config': self.config
            }

            # 保存文件
            torch.save(checkpoint, filepath)
            print(f"💾 模型已保存至：{filepath}")

        except Exception as e:
            print(f"❌ 保存模型失败：{e}")

    def load_models(self, filepath):
        """
        加载模型
        :param filepath: 模型路径
        :return: 是否加载成功
        """
        try:
            if not os.path.exists(filepath):
                print(f"❌ 模型文件不存在：{filepath}")
                return False

            # 加载检查点
            checkpoint = torch.load(
                filepath,
                map_location=self.device,
            )

            # 加载网络参数
            for i in range(self.num_agents):
                self.actors[i].load_state_dict(checkpoint['actors'][i])
                self.actor_targets[i].load_state_dict(checkpoint['actor_targets'][i])
                self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][i])

            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

            # 加载训练状态
            self.training_step = checkpoint.get('training_step', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.noise_scale = checkpoint.get('noise_scale', self.config['initial_noise_scale'])

            # 加载OU噪声状态
            if 'ou_noise_states' in checkpoint:
                for i, noise_state in enumerate(checkpoint['ou_noise_states']):
                    if i < len(self.ou_noises):
                        self.ou_noises[i].state = noise_state

            # 加载配置（覆盖当前配置）
            if 'config' in checkpoint:
                self.config.update(checkpoint['config'])

            print(f"✅ 模型加载成功：{filepath}")
            print(f"   训练步数：{self.training_step}，当前回合：{self.episode_count}")
            print(f"   噪声缩放因子：{self.noise_scale:.3f}")

            return True

        except Exception as e:
            print(f"❌ 加载模型失败：{e}")
            import traceback
            traceback.print_exc()
            return False

    def get_training_status(self):
        """获取当前训练状态"""
        buffer_info = self.replay_buffer.get_info()
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'noise_scale': self.noise_scale,
            'buffer_info': buffer_info,
            'last_losses': self.training_losses[-10:] if len(self.training_losses) >= 10 else self.training_losses
        }

    def print_training_status(self):
        """打印训练状态摘要"""
        status = self.get_training_status()
        print("\n" + "="*60)
        print("📋 CTDE-DDPG训练状态摘要")
        print("="*60)
        print(f"训练步数：{status['training_step']}")
        print(f"已训练回合：{status['episode_count']}")
        print(f"噪声缩放因子：{status['noise_scale']:.3f}")
        print(f"经验缓冲区：{status['buffer_info']['current_size']}/{status['buffer_info']['capacity']} "
              f"（使用率：{status['buffer_info']['usage_rate']:.1f}%）")
        print(f"优先级回放beta：{status['buffer_info']['current_beta']:.3f}")

        if status['last_losses']:
            avg_critic_loss = np.mean([l['critic_loss'] for l in status['last_losses']])
            avg_actor_loss = np.mean([l['actor_loss'] for l in status['last_losses']])
            avg_q_value = np.mean([l['avg_q_value'] for l in status['last_losses']])
            print(f"最近10步平均损失：")
            print(f"  Critic损失：{avg_critic_loss:.4f}")
            print(f"  Actor损失：{avg_actor_loss:.4f}")
            print(f"  平均Q值：{avg_q_value:.3f}")

        print("="*60 + "\n")

# 测试代码（验证算法完整性）
if __name__ == "__main__":
    def test_ctde_ddpg():
        """测试CTDE-DDPG算法核心功能"""
        print("=== CTDE-DDPG算法完整测试 ===")

        # 初始化算法
        agent = CTDE_DDPG(
            num_agents=3,
            state_dim=9,
            action_dim=1
        )

        try:
            # 1. 测试动作生成
            print("\n1. 测试动作生成...")
            test_states = [np.random.randn(9) for _ in range(3)]
            actions_explore = agent.get_actions(test_states, exploration=True)
            actions_no_explore = agent.get_actions(test_states, exploration=False)

            print(f"   带探索动作：{[f'{a[0]:.3f}' for a in actions_explore]}")
            print(f"   无探索动作：{[f'{a[0]:.3f}' for a in actions_no_explore]}")
            assert all(-1.0 <= a[0] <= 1.0 for a in actions_explore), "动作超出[-1,1]范围"

            # 2. 测试噪声重置和衰减
            print("\n2. 测试OU噪声...")
            agent.reset_noise()  # 测试reset_noise方法
            initial_noise_scale = agent.noise_scale
            agent.decay_noise()
            assert agent.noise_scale < initial_noise_scale, "噪声衰减失败"
            print(f"   噪声衰减测试通过（{initial_noise_scale:.3f} → {agent.noise_scale:.3f}）")

            # 3. 测试经验回放
            print("\n3. 测试经验回放...")
            for _ in range(200):
                state = [np.random.randn(9) for _ in range(3)]
                action = [np.random.uniform(-1, 1, 1) for _ in range(3)]
                reward = np.random.uniform(-1, 1, 3)
                next_state = [np.random.randn(9) for _ in range(3)]
                done = [False]*3
                agent.replay_buffer.push(state, action, reward, next_state, done)

            assert len(agent.replay_buffer) == 200, "经验回放缓冲区添加失败"
            print(f"   经验缓冲区填充成功（当前大小：{len(agent.replay_buffer)}）")

            # 4. 测试网络更新
            print("\n4. 测试网络更新...")
            for _ in range(5):
                loss_info = agent.update()
                if loss_info:
                    print(f"   第{_+1}次更新：Critic损失={loss_info['critic_loss']:.4f}，Actor损失={loss_info['actor_loss']:.4f}")

            assert agent.training_step > 0, "网络更新失败"
            print("   网络更新测试通过")

            # 5. 测试模型保存和加载
            print("\n5. 测试模型保存与加载...")
            save_path = "test_model.pth"
            agent.save_models(save_path)  # 测试save_models方法

            # 创建新实例并加载
            agent2 = CTDE_DDPG(num_agents=3, state_dim=9, action_dim=1)
            load_success = agent2.load_models(save_path)
            assert load_success, "模型加载失败"

            # 验证加载后的动作一致性
            test_states2 = [np.random.randn(9) for _ in range(3)]
            actions1 = agent.get_actions(test_states2, exploration=False)
            actions2 = agent2.get_actions(test_states2, exploration=False)
            action_diff = [abs(a1[0] - a2[0]) for a1, a2 in zip(actions1, actions2)]

            assert max(action_diff) < 1e-5, f"模型加载后动作不一致（最大差异：{max(action_diff)}）"
            print("   模型保存与加载测试通过")

            # 6. 打印训练状态
            print("\n6. 训练状态摘要...")
            agent.print_training_status()

            # 清理测试文件
            if os.path.exists(save_path):
                os.remove(save_path)

            print("\n✅ CTDE-DDPG算法测试全部通过！")
            return True

        except Exception as e:
            print(f"\n❌ 测试失败：{e}")
            import traceback
            traceback.print_exc()
            return False

    # 运行测试
    test_ctde_ddpg()