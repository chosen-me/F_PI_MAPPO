import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==========================================
# 共享经验回放池
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 1. DQN 算法 (修复状态归一化)
# ==========================================
class DQN_Agent:
    def __init__(self, num_agents, state_dim, action_dim=5):
        self.num_agents = num_agents
        self.action_space = [-3.0, -1.0, 0.0, 1.0, 3.0]
        self.action_dim = action_dim

        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.action_dim)
        ).to(device)
        self.target_q_net = copy.deepcopy(self.q_net)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.memory = ReplayBuffer()

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.update_step = 0

        # 【新增：状态归一化因子】
        self.state_scale = torch.FloatTensor([30.0, 3.0, 50.0, 10.0, 100.0, 10.0, 1.0, 3.0, 3.0]).to(device)

    def select_action(self, states, exploration=True):
        actions_out = []
        if exploration and random.random() < self.epsilon:
            for _ in states:
                action_idx = random.randint(0, self.action_dim - 1)
                actions_out.append([self.action_space[action_idx] / 3.0])
        else:
            with torch.no_grad():
                for state in states:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    # 【应用归一化】
                    state_tensor = state_tensor / self.state_scale
                    q_values = self.q_net(state_tensor)
                    action_idx = torch.argmax(q_values).item()
                    actions_out.append([self.action_space[action_idx] / 3.0])
        return actions_out

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 【应用归一化】
        states = torch.FloatTensor(states).to(device) / self.state_scale
        next_states = torch.FloatTensor(next_states).to(device) / self.state_scale

        action_indices = []
        for a in actions:
            val = a[0] * 3.0
            idx = min(range(len(self.action_space)), key=lambda i: abs(self.action_space[i] - val))
            action_indices.append(idx)
        actions = torch.LongTensor(action_indices).unsqueeze(1).to(device)

        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_q_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % 100 == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save_models(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_models(self, path):
        self.q_net.load_state_dict(torch.load(path))


# ==========================================
# 2. DDPG 算法 (修复状态归一化)
# ==========================================
class DDPG_Agent:
    def __init__(self, num_agents, state_dim, action_dim=1):
        self.num_agents = num_agents

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.memory = ReplayBuffer()

        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_std = 0.2

        # 【新增：状态归一化因子】
        self.state_scale = torch.FloatTensor([30.0, 3.0, 50.0, 10.0, 100.0, 10.0, 1.0, 3.0, 3.0]).to(device)

    def select_action(self, states, exploration=True):
        actions_out = []
        with torch.no_grad():
            for state in states:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                # 【应用归一化】
                state_tensor = state_tensor / self.state_scale

                action = self.actor(state_tensor).cpu().numpy()[0]

                if exploration:
                    noise = np.random.normal(0, self.noise_std, size=action.shape)
                    action = np.clip(action + noise, -1.0, 1.0)

                actions_out.append(action.tolist())
        return actions_out

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 【应用归一化】
        states = torch.FloatTensor(states).to(device) / self.state_scale
        next_states = torch.FloatTensor(next_states).to(device) / self.state_scale

        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # 1. 更新 Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(torch.cat([next_states, next_actions], 1))
            target_q = rewards + (self.gamma * target_q * (1 - dones))

        current_q = self.critic(torch.cat([states, actions], 1))
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. 更新 Actor
        actor_loss = -self.critic(torch.cat([states, self.actor(states)], 1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. 目标网络软更新
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return actor_loss.item(), critic_loss.item()

    def save_models(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def load_models(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


import numpy as np
from scipy.optimize import minimize


# ==========================================
# 传统控制策略：基于常数车头时距 (CTH) 的 CACC
# 参考文献: Naus G J L, et al. "String-stable CACC design and experimental validation", IEEE TVT, 2010.
# ==========================================
class Traditional_CACC:
    def __init__(self):
        # 严格遵循频域分析证明绝对串稳定的增益参数
        self.kp = 0.45
        self.kd = 0.25
        # CTH 策略核心参数：车速越快，期望间距越大
        self.time_headway = 0.6
        self.standstill_distance = 5.0

    def select_action(self, states, exploration=False):
        actions = []
        for state in states:
            v_ego = state[0]
            spacing = state[2]
            rel_speed = state[3]
            front_accel = state[7]

            # 动态期望间距 (保障绝对安全，但效率不如固定间距)
            desired_spacing = self.standstill_distance + self.time_headway * v_ego
            spacing_error = spacing - desired_spacing

            # 完美的前馈与 PD 反馈结合
            accel = front_accel + self.kp * spacing_error + self.kd * rel_speed

            # 物理安全边界：防止环境异常数据注入
            if spacing < 6.0:
                accel = -3.0

            actions.append([np.clip(accel / 3.0, -1.0, 1.0)])
        return actions


# # ==========================================
# # 线性模型预测控制 (Linear MPC, 引入 CBF 机制)
# # 参考文献: Zheng Y, et al. "Distributed model predictive control for heterogeneous vehicle platoons", IEEE TCST, 2016.
# # ==========================================
# class MPC_CACC:
#     def __init__(self):
#         self.dt = 0.1
#         self.H = 5  # 将预测时域锁定为 5，保证 L-BFGS-B 的实时求解成功率
#         self.desired_spacing = 15.0
#
#         self.w_spacing = 1.0
#         self.w_speed = 0.5
#         self.w_accel = 0.1
#         self.w_jerk = 0.1
#
#         # 热启动记忆库
#         self.U_prev = [np.zeros(self.H) for _ in range(3)]
#
#     def mpc_cost_function(self, U, *args):
#         spacing, rel_speed, front_accel, prev_u = args
#         cost = 0.0
#         s, v, u_last = spacing, rel_speed, prev_u
#
#         for i in range(self.H):
#             u = U[i]
#             s = s + v * self.dt
#             v = v + (front_accel - u) * self.dt
#             jerk = (u - u_last) / self.dt
#
#             # 【核心护城河】：控制障碍函数 (CBF)
#             # 预测时域内只要可能小于 8 米，赋予毁灭性代价值，逼迫优化器无条件减速
#             if s < 8.0:
#                 cost += 50000.0 * (8.0 - s) ** 2
#
#             cost += self.w_spacing * (s - self.desired_spacing) ** 2
#             cost += self.w_speed * (v) ** 2
#             cost += self.w_accel * (u) ** 2
#             cost += self.w_jerk * (jerk) ** 2
#
#             u_last = u
#
#         return cost
#
#     def select_action(self, states, exploration=False):
#         actions = []
#         for i, state in enumerate(states):
#             spacing = state[2]
#             rel_speed = state[3]
#             front_accel = state[7]
#             current_accel = state[1]
#
#             # 极端紧急制动：接管优化器
#             if spacing < 5.0:
#                 actions.append([-1.0])
#                 self.U_prev[i] = np.zeros(self.H)
#                 continue
#
#             args = (spacing, rel_speed, front_accel, current_accel)
#
#             # 热启动：平移上次解作为初始猜测
#             U0 = np.roll(self.U_prev[i], -1)
#             U0[-1] = U0[-2]
#             bounds = [(-3.0, 3.0)] * self.H
#
#             # 【鲁棒求解器】：L-BFGS-B 处理边界约束的二次型问题极度稳定
#             res = minimize(self.mpc_cost_function, U0, args=args, bounds=bounds, method='L-BFGS-B')
#
#             if res.success:
#                 opt_accel = res.x[0]
#                 self.U_prev[i] = res.x
#             else:
#                 opt_accel = 0.45 * (spacing - 15.0) + 0.25 * rel_speed
#
#             actions.append([np.clip(opt_accel / 3.0, -1.0, 1.0)])
#         return actions

# import numpy as np
# from scipy.optimize import minimize
#
#
# # ==========================================
# # 分布式模型预测控制 (Linear MPC)
# # 引入: 非对称代价函数 (Asymmetric Cost) & TTC 安全包络 (AEB Envelope)
# # 参考文献: Zheng Y, et al. (2016) 基础框架 + 工业界标准 AEB 兜底策略
# # ==========================================
# class MPC_CACC:
#     def __init__(self):
#         self.dt = 0.1
#         self.H = 10  # 【修复1】：恢复 1.0秒 的长预测视距，让 MPC 看得更远
#         self.desired_spacing = 15.0
#
#         self.w_spacing = 1.0
#         self.w_speed = 0.5
#         self.w_accel = 0.1
#         self.w_jerk = 0.1
#
#         self.U_prev = [np.zeros(self.H) for _ in range(3)]
#
#     def mpc_cost_function(self, U, *args):
#         spacing, rel_speed, front_accel, prev_u = args
#         cost = 0.0
#         s, v, u_last = spacing, rel_speed, prev_u
#
#         for i in range(self.H):
#             u = U[i]
#             s = s + v * self.dt
#             v = v + (front_accel - u) * self.dt
#             jerk = (u - u_last) / self.dt
#
#             # 【核心修复2：非对称代价函数】
#             # 车距如果小于期望值 (15m)，给予 10 倍的惩罚权重！宁愿跟得远，绝对不靠近！
#             if s < self.desired_spacing:
#                 cost += 10.0 * self.w_spacing * (self.desired_spacing - s) ** 2
#             else:
#                 cost += self.w_spacing * (s - self.desired_spacing) ** 2
#
#             # 控制障碍函数 CBF：预测到车距极度危险时，给予毁灭性惩罚
#             if s < 8.0:
#                 cost += 100000.0 * (8.0 - s) ** 2
#
#             cost += self.w_speed * (v) ** 2
#             cost += self.w_accel * (u) ** 2
#             cost += self.w_jerk * (jerk) ** 2
#
#             u_last = u
#
#         return cost
#
#     def select_action(self, states, exploration=False):
#         actions = []
#         for i, state in enumerate(states):
#             spacing = state[2]
#             rel_speed = state[3]
#             front_accel = state[7]
#             current_accel = state[1]
#
#             # 【核心修复3：基于 Euro NCAP 标准的 TTC 安全包络 (AEB 机制)】
#             # 计算碰撞时间 (Time-To-Collision)
#             ttc = spacing / abs(rel_speed) if rel_speed < -0.1 else 999.0
#
#             # 如果车距小于 8m，或 TTC 小于 2.5s，强制切断 MPC，激活 AEB 紧急制动
#             if spacing < 8.0 or ttc < 2.5:
#                 actions.append([-1.0])  # 满负荷紧急刹车 (-3.0 m/s^2)
#                 self.U_prev[i] = np.zeros(self.H)
#                 continue
#
#             args = (spacing, rel_speed, front_accel, current_accel)
#             U0 = np.roll(self.U_prev[i], -1)
#             U0[-1] = U0[-2]
#             bounds = [(-3.0, 3.0)] * self.H
#
#             res = minimize(self.mpc_cost_function, U0, args=args, bounds=bounds, method='L-BFGS-B')
#
#             if res.success:
#                 opt_accel = res.x[0]
#                 self.U_prev[i] = res.x
#             else:
#                 # 优化器偶发死锁时的 PD 降级策略
#                 opt_accel = 0.45 * (spacing - 15.0) + 0.25 * rel_speed
#
#             actions.append([np.clip(opt_accel / 3.0, -1.0, 1.0)])
#         return actions

import numpy as np
from scipy.optimize import minimize


# ==========================================
# 动态车头时距模型预测控制 (CTH-MPC)
# 引入: 常数车头时距 (Constant Time Headway) & 动态安全包络
# 参考文献: Zheng Y, et al. (2016) 结合现代 ACC 的 CTH 安全策略
# ==========================================
class MPC_CACC:
    def __init__(self):
        self.dt = 0.1
        self.H = 10

        # 【核心修复 1】：引入 CTH 策略参数
        self.time_headway = 0.8  # 0.8秒的车头时距，保证高速跟驰的绝对安全
        self.standstill_distance = 5.0  # 绝对停止距离 5.0 米

        self.w_spacing = 1.0
        self.w_speed = 0.5
        self.w_accel = 0.1
        self.w_jerk = 0.1

        self.U_prev = [np.zeros(self.H) for _ in range(3)]

    def mpc_cost_function(self, U, *args):
        # 相比之前，额外传入了自车速度 v_ego
        spacing, rel_speed, front_accel, prev_u, v_ego = args
        cost = 0.0
        s, v_rel, u_last, v_e = spacing, rel_speed, prev_u, v_ego

        for i in range(self.H):
            u = U[i]
            # 状态演化预测
            s = s + v_rel * self.dt
            v_rel = v_rel + (front_accel - u) * self.dt
            v_e = v_e + u * self.dt  # 预测自车未来的速度
            jerk = (u - u_last) / self.dt

            # 【核心修复 2：动态计算期望间距】
            # 车速越快，期望间距越大，从数学模型根源上消除追尾风险
            desired_spacing = self.standstill_distance + self.time_headway * v_e

            # 非对称代价：严厉惩罚小于期望间距的行为
            if s < desired_spacing:
                cost += 10.0 * self.w_spacing * (desired_spacing - s) ** 2
            else:
                cost += self.w_spacing * (s - desired_spacing) ** 2

            # 控制障碍函数 CBF：绝对物理界限
            if s < 6.0:
                cost += 100000.0 * (6.0 - s) ** 2

            cost += self.w_speed * (v_rel) ** 2
            cost += self.w_accel * (u) ** 2
            cost += self.w_jerk * (jerk) ** 2

            u_last = u

        return cost

    def select_action(self, states, exploration=False):
        actions = []
        for i, state in enumerate(states):
            v_ego = state[0]  # 提取自车速度
            spacing = state[2]
            rel_speed = state[3]
            front_accel = state[7]
            current_accel = state[1]

            # 【核心修复 3：动态 AEB 紧急制动包络】
            # 安全阈值随速度动态变化，速度越快，越早踩死刹车
            safe_threshold = self.standstill_distance + 0.4 * v_ego
            ttc = spacing / abs(rel_speed) if rel_speed < -0.1 else 999.0

            if spacing < safe_threshold or ttc < 2.5:
                actions.append([-1.0])
                self.U_prev[i] = np.zeros(self.H)
                continue

            # 正常 MPC 优化求解
            args = (spacing, rel_speed, front_accel, current_accel, v_ego)
            U0 = np.roll(self.U_prev[i], -1)
            U0[-1] = U0[-2]
            bounds = [(-3.0, 3.0)] * self.H

            res = minimize(self.mpc_cost_function, U0, args=args, bounds=bounds, method='L-BFGS-B')

            if res.success:
                opt_accel = res.x[0]
                self.U_prev[i] = res.x
            else:
                # 优化器偶发死锁时的降级方案：CTH PD 控制
                desired_spacing = self.standstill_distance + self.time_headway * v_ego
                opt_accel = 0.45 * (spacing - desired_spacing) + 0.25 * rel_speed

            actions.append([np.clip(opt_accel / 3.0, -1.0, 1.0)])
        return actions