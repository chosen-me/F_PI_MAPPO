import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os

# 检查设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.global_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.global_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出限制在 [-1, 1] 之间
        )
        # 【修复3：降低初始探索噪声】将初始对数标准差设为-0.5 (即std≈0.6)，避免瞎探索
        self.action_log_std = nn.Parameter(torch.full((1, action_dim), -0.5))

    def forward(self, state):
        action_mean = self.net(state)
        action_std = self.action_log_std.exp().expand_as(action_mean)
        return action_mean, action_std


class Critic(nn.Module):
    def __init__(self, global_state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_state):
        return self.net(global_state)


class MAPPO:
    def __init__(self, num_agents, state_dim, action_dim, config=None):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = state_dim * num_agents

        # 默认配置
        self.config = {
            'lr_actor': 1e-4,  # 稍微调低学习率，求稳
            'lr_critic': 3e-4,
            'gamma': 0.99,
            'K_epochs': 10,
            'eps_clip': 0.2,
            'entropy_coef': 0.01,
        }
        if config:
            self.config.update(config)

        # 策略网络和价值网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(self.global_state_dim).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.config['lr_actor'])
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.config['lr_critic'])
        self.mse_loss = nn.MSELoss()

        self.buffer = RolloutBuffer()
        self.training_step = 0

        # 【修复1：状态归一化因子】将9维状态除以它们的物理极限值，压缩到 [-1, 1] 附近
        # 顺序: 速度(30), 加速度(3), 间距(50), 相对速度(10), 与领头车间距(100), 领头车速差(10), 曲率(1), 前车加(3), 领车加(3)
        self.state_scale = torch.FloatTensor([30.0, 3.0, 50.0, 10.0, 100.0, 10.0, 1.0, 3.0, 3.0]).to(device)

    def select_action(self, states, exploration=True):
        """选择动作，states为一个列表，包含所有agent的9维状态"""
        with torch.no_grad():
            # 【应用修复1】在这里直接除以缩放因子，之后整个网络内部都是健康的小数值
            states_tensor = torch.FloatTensor(np.array(states)).to(device) / self.state_scale

            global_state = states_tensor.view(-1).unsqueeze(0)

            action_mean, action_std = self.actor(states_tensor)
            dist = Normal(action_mean, action_std)

            if exploration:
                action = dist.sample()
            else:
                action = action_mean

            action_logprob = dist.log_prob(action).sum(dim=-1)
            state_val = self.critic(global_state)

        # 记录到buffer (记录的是归一化后的状态)
        self.buffer.states.append(states_tensor)
        self.buffer.global_states.append(global_state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.values.append(state_val)

        # 确保动作在 [-1, 1] 之间，然后转为numpy
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy().tolist()

    def update(self):
        """执行PPO网络更新"""
        # --- 1. 计算折扣奖励 (Returns) ---
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            team_reward = sum(reward) / self.num_agents
            discounted_reward = team_reward + (self.config['gamma'] * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 【修复2：绝对禁止归一化Returns，只转成Tensor】
        returns = torch.tensor(rewards, dtype=torch.float32).to(device)

        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
        old_global_states = torch.stack(self.buffer.global_states, dim=0).view(-1, self.global_state_dim).detach().to(
            device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        old_values = torch.stack(self.buffer.values, dim=0).view(-1).detach().to(device)

        # --- 2. 计算优势函数 (Advantages) 并进行归一化 ---
        advantages = returns - old_values
        # 【应用修复2】在这里归一化优势函数，保证Actor更新的稳定性，同时Critic依然能学到真实的分数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.detach()

        # --- 3. PPO 迭代更新 ---
        actor_loss_history = []
        critic_loss_history = []

        for _ in range(self.config['K_epochs']):
            action_mean, action_std = self.actor(old_states)
            dist = Normal(action_mean, action_std)

            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)

            state_values = self.critic(old_global_states).view(-1)

            ratios = torch.exp(logprobs - old_logprobs)

            flat_ratios = ratios.view(-1)
            flat_advantages = advantages.unsqueeze(-1).expand(-1, self.num_agents).reshape(-1)

            surr1 = flat_ratios * flat_advantages
            surr2 = torch.clamp(flat_ratios, 1 - self.config['eps_clip'], 1 + self.config['eps_clip']) * flat_advantages

            # 演员网络 Loss
            loss_actor = -torch.min(surr1, surr2).mean() - self.config['entropy_coef'] * dist_entropy.mean()

            # 评论家网络 Loss (现在目标 returns 包含了真实分数尺度，Critic将稳定收敛)
            loss_critic = self.mse_loss(state_values, returns)

            # 反向传播更新
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

            actor_loss_history.append(loss_actor.item())
            critic_loss_history.append(loss_critic.item())

        self.training_step += 1
        self.buffer.clear()

        return np.mean(actor_loss_history), np.mean(critic_loss_history)

    def save_models(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load_models(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            return True
        return False