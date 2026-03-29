import numpy as np
import time
import os
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# 导入 TensorBoard (PyTorch自带)
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入环境和MAPPO算法
from cacc_real_time_env import CACCRealTimeEnv
from pi_mappo import PI_MAPPO


# ==========================================
# 强化学习专用自动化早停机制
# ==========================================
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1.0, window_size=20, min_episodes=250):
        """
        :param patience: 容忍多少个回合滑动平均没有提升
        :param min_delta: 提升的最小判定阈值
        :param window_size: 滑动平均的窗口大小
        :param min_episodes: 最少训练回合数
        """
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.min_episodes = min_episodes

        self.best_ma = -np.inf  # 历史最佳滑动平均分
        self.counter = 0  # 失去耐心的计数器
        self.reward_history = []

    def check(self, episode_reward, episode):
        self.reward_history.append(episode_reward)

        # 必须满足最小训练回合数，且历史记录足够计算均值
        if episode < self.min_episodes or len(self.reward_history) < self.window_size:
            return False, 0.0

        # 计算最近 window_size 个回合的滑动平均奖励
        current_ma = np.mean(self.reward_history[-self.window_size:])

        # 判断是否有实质性提升
        if current_ma > self.best_ma + self.min_delta:
            self.best_ma = current_ma
            self.counter = 0  # 表现有提升，重置耐心计数器
        else:
            self.counter += 1  # 表现未提升，增加失去耐心值

        # 判断是否触发早停
        should_stop = self.counter >= self.patience
        return should_stop, current_ma


# ==========================================
# CACC MAPPO 训练主控类
# ==========================================
class CACC_MAPPO_Trainer:
    def __init__(self, config_path=None):
        self.config = self.get_default_config()
        self.training_start_time = datetime.now()
        self.training_dir = self._create_training_dir()

        self.env = CACCRealTimeEnv(self.config['environment'])
        self.agent = PI_MAPPO(
            num_agents=3,
            state_dim=9,
            action_dim=1,
            config=self.config['algorithm']
        )

        # 记录训练数据
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.best_reward = -float('inf')

        self.log_file = os.path.join(self.training_dir, "training.log")

        # 初始化 TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.training_dir, 'tensorboard'))

        # 初始化 Matplotlib 动态图表
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.canvas.manager.set_window_title('PIMAPPO 实时训练监控')

        # 初始化早停机制 (容忍50回合不提升，最小训练250回合)
        self.early_stopping = EarlyStopping(patience=50, min_delta=0.5, window_size=20, min_episodes=250)

    def get_default_config(self):
        return {
            'environment': {
                'sumo_config': 'straight.sumocfg',
                'gui': False,
                'step_length': 0.1,
                'collision_check': True,
                'max_steps_per_episode': 500,
                'reward_weights': {
                    'spacing': 1.0, 'speed': 1.0, 'acceleration': 0.1,
                    'jerk': 0.1, 'collision': 10.0
                }
            },
            'algorithm': {
                'lr_actor': 1e-4,
                'lr_critic': 3e-4,
                'gamma': 0.99,
                'K_epochs': 10,
                'eps_clip': 0.2,
                'entropy_coef': 0.01,
            },
            'training': {
                'num_episodes': 1500,
                'save_interval': 50,
                'plot_interval': 10
            }
        }

    def _create_training_dir(self):
        timestamp = self.training_start_time.strftime('%Y%m%d_%H%M%S')
        training_dir = f"pimappo_results_{timestamp}"
        for subdir in ['models', 'logs', 'plots', 'data', 'tensorboard']:
            os.makedirs(os.path.join(training_dir, subdir), exist_ok=True)
        return training_dir

    def _log_message(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.rstrip())
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def live_plot(self, episode):
        """实时更新并绘制图表"""
        for ax in self.axs:
            ax.cla()

        episodes = range(1, len(self.episode_rewards) + 1)

        # 1. 绘制团队总奖励
        self.axs[0].plot(episodes, self.episode_rewards, 'b-', alpha=0.4, label='回合奖励')
        window = min(10, len(self.episode_rewards))
        if window > 1:
            rewards_ma = np.convolve(self.episode_rewards, np.ones(window) / window, mode='valid')
            self.axs[0].plot(range(window, len(self.episode_rewards) + 1), rewards_ma, 'r-', linewidth=2,
                             label=f'滑动平均 ({window})')
        self.axs[0].set_title(f'团队总奖励 (当前回合: {episode})')
        self.axs[0].set_ylabel('Reward')
        self.axs[0].legend()
        self.axs[0].grid(True, alpha=0.3)

        # 2. 绘制 Actor Loss
        self.axs[1].plot(episodes, self.actor_losses, 'g-', label='Actor Loss')
        self.axs[1].set_title('策略网络损失 (Actor Loss)')
        self.axs[1].set_ylabel('Loss')
        self.axs[1].legend()
        self.axs[1].grid(True, alpha=0.3)

        # 3. 绘制 Critic Loss
        self.axs[2].plot(episodes, self.critic_losses, 'm-', label='Critic Loss')
        self.axs[2].set_title('价值网络损失 (Critic Loss)')
        self.axs[2].set_xlabel('回合 (Episode)')
        self.axs[2].set_ylabel('Loss')
        self.axs[2].legend()
        self.axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

        plot_path = os.path.join(self.training_dir, 'plots', 'live_training_progress.png')
        plt.savefig(plot_path)

    # ==========================================
    # 核心 Train 函数 (包含早停检测)
    # ==========================================
    def train(self):
        self._log_message("🚀 开始基于MAPPO的CACC车辆编队训练！")
        start_time = time.time()

        for episode in range(1, self.config['training']['num_episodes'] + 1):
            episode_start_time = time.time()
            states = self.env.reset()
            episode_reward = 0
            done = False
            step = 0

            pbar = tqdm(range(self.config['environment']['max_steps_per_episode']),
                        desc=f"回合 {episode:3d}/{self.config['training']['num_episodes']}",
                        leave=False)

            while not done and step < self.config['environment']['max_steps_per_episode']:
                actions = self.agent.select_action(states, exploration=True)
                next_states, rewards, done, info = self.env.step(actions)

                self.agent.buffer.rewards.append(rewards)
                self.agent.buffer.is_terminals.append(done)

                states = next_states
                episode_reward += sum(rewards)
                step += 1
                pbar.update(1)

            pbar.close()

            # PPO 网络更新
            actor_loss, critic_loss = self.agent.update()

            # 记录本回合数据
            self.episode_rewards.append(episode_reward)
            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)

            # 将数据写入 TensorBoard
            self.writer.add_scalar('Reward/Total_Team_Reward', episode_reward, episode)
            self.writer.add_scalar('Loss/Actor_Loss', actor_loss, episode)
            self.writer.add_scalar('Loss/Critic_Loss', critic_loss, episode)

            self._log_message(
                f"回合 {episode:3d} 完成 | 奖励：{episode_reward:7.2f} | "
                f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}"
            )

            # 刷新实时动态图表
            if episode % self.config['training']['plot_interval'] == 0:
                self.live_plot(episode)

            # 保存最佳模型和检查点
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                best_model_path = os.path.join(self.training_dir, 'models', 'best_pimappo_3.pth')
                self.agent.save_models(best_model_path)
                self._log_message(f"🎉 发现最佳模型！奖励提升至：{episode_reward:.2f}")

            if episode % self.config['training']['save_interval'] == 0:
                checkpoint_path = os.path.join(self.training_dir, 'models', f'checkpoint_ep{episode:04d}.pth')
                self.agent.save_models(checkpoint_path)

            # ==========================================
            # 自动化早停检测核心逻辑
            # ==========================================
            should_stop, current_ma = self.early_stopping.check(episode_reward, episode)

            # 在日志中打印早停的监控状态
            if episode >= self.early_stopping.min_episodes:
                self._log_message(
                    f"   [早停监控] 滑动平均奖励: {current_ma:.2f} | 历史最佳: {self.early_stopping.best_ma:.2f} | 失去耐心: {self.early_stopping.counter}/{self.early_stopping.patience}")

            if should_stop:
                self._log_message("=" * 60)
                self._log_message(f"🛑 自动早停触发！")
                self._log_message(f"连续 {self.early_stopping.patience} 个回合模型平均表现没有实质提升。")
                self._log_message(f"🏆 最终稳定滑动平均奖励为: {current_ma:.2f} (在回合 {episode} 终止)")
                self._log_message("=" * 60)
                break  # 跳出训练大循环

        self.env.close()
        self.writer.close()
        self._log_message("🏁 训练结束！")

        # 训练结束后保持图表窗口不关闭
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    trainer = CACC_MAPPO_Trainer()
    trainer.train()