import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from math import pi

# 导入你的环境和算法
from cacc_real_time_env import CACCRealTimeEnv
from mappo import MAPPO          # 导入老版 MAPPO
from pi_mappo import    PI_MAPPO      # 导入你创新的 A-MAPPO
from baseline_agents import DQN_Agent, DDPG_Agent, Traditional_CACC, MPC_CACC

# 设置中文字体与全局绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei', rc={'axes.unicode_minus': False})

# ... 前面的导入保持不变 ...
from cacc_real_time_env import CACCRealTimeEnv
from mappo import MAPPO  # 导入老版 MAPPO
from pi_mappo import PI_MAPPO  # 导入你创新的 A-MAPPO
from baseline_agents import DQN_Agent, DDPG_Agent, Traditional_CACC, MPC_CACC


class CACCEvaluator:
    def __init__(self, episodes_per_scenario=100):
        self.episodes_per_scenario = episodes_per_scenario
        self.scenarios = ['straight', 'curve_left', 'curve_right']
        # 【修改1：将 A-MAPPO 加入评估列表】
        self.algorithms = ['Traditional_CACC', 'MPC_CACC', 'DQN', 'DDPG', 'MAPPO', 'PI-MAPPO']

        self.out_dir = "evaluation_results_off"
        self.data_dir = os.path.join(self.out_dir, "data")
        self.plot_dir = os.path.join(self.out_dir, "plots")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # 【修改2：在代理字典中实例化 A-MAPPO】
        self.agents = {
            'Traditional_CACC': Traditional_CACC(),
            'MPC_CACC': MPC_CACC(),
            'DQN': DQN_Agent(3, 9),
            'DDPG': DDPG_Agent(3, 9),
            'MAPPO': MAPPO(3, 9, 1),
            'PI-MAPPO': PI_MAPPO(3, 9, 1)  # 你的创新算法
        }

        # 尝试加载训练好的权重
        # 请确保你的 models/ 文件夹下有 best_mappo.pth 和 best_pimappo_2.pth
        for algo in ['DQN', 'DDPG', 'MAPPO']:
            model_path = f'models/best_{algo.lower()}.pth'
            if os.path.exists(model_path):
                self.agents[algo].load_models(model_path)

        # 单独加载 A-MAPPO
        if os.path.exists('models/best_pimappo.pth'):
            self.agents['PI-MAPPO'].load_models('models/best_pimappo.pth')
            print("🌟 成功加载创新版 PI-MAPPO 模型！")

        self.all_metrics_records = []


    def run_evaluation(self):
        """执行全方位评估并收集环境原生数据"""
        for scenario in self.scenarios:
            print(f"\n" + "=" * 50)
            print(f"🚗 开始评估场景: {scenario.upper()}")
            print("=" * 50)

            env_config = {
                'scenario_type': scenario,
                'gui': False,
                'max_steps_per_episode': 500
            }
            env = CACCRealTimeEnv(env_config)

            for algo_name in self.algorithms:
                print(f"  ▶ 正在运行算法: {algo_name}")
                agent = self.agents[algo_name]

                for ep in tqdm(range(self.episodes_per_scenario), leave=False, desc=f"{algo_name} 测试中"):
                    states = env.reset()
                    done = False
                    step = 0

                    while not done and step < env_config['max_steps_per_episode']:
                        # 关闭探索
                        actions = agent.select_action(states, exploration=False) if hasattr(agent,
                                                                                            'select_action') else agent.get_actions(
                            states, exploration=False)
                        states, _, done, _ = env.step(actions)
                        step += 1

                    # 1. 提取环境内置的丰富评价指标
                    metrics = env.get_performance_metrics()
                    if not metrics:
                        continue

                    # 记录均值数据用于后续绘图与统计 (对3辆跟随车取平均)
                    record = {
                        'Scenario': scenario,
                        'Algorithm': algo_name,
                        'Episode': ep,
                        'Collision_Count': metrics['collision_count'],
                        'Avg_Speed': np.mean(metrics['avg_speed']),
                        'Speed_Error': np.mean(metrics['speed_following_error']),
                        'Min_Spacing': np.min(metrics['min_spacing']),
                        'Spacing_Error': np.mean(metrics['spacing_error']),
                        'Spacing_Std': np.mean(metrics['spacing_std']),
                        'TTC_Below_3s_Pct': np.mean(metrics['ttc_below_3_percent']),
                        'Accel_RMS': np.mean(metrics['accel_rms']),
                        'Avg_Jerk': np.mean(metrics['avg_jerk'])
                    }
                    self.all_metrics_records.append(record)

                    # 2. 保存极具价值的单回合微观时序数据 (.npz)
                    if ep == 0:  # 每个场景每个算法只保存第一回合的时序数据用于画细节图
                        save_path = os.path.join(self.data_dir, f"{scenario}_{algo_name}_ep{ep}.npz")
                        env.save_episode_data(filename=save_path)

            env.close()

        # 3. 数据汇总落盘
        self.df_metrics = pd.DataFrame(self.all_metrics_records)
        csv_path = os.path.join(self.data_dir, "comprehensive_metrics_report.csv")
        self.df_metrics.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 所有量化指标已保存至: {csv_path}")

        # 4. 生成论文图表
        self.generate_paper_charts()

    # ==========================================
    # 论文制图系统 (高斯平滑、箱线图、雷达图)
    # ==========================================
    def generate_paper_charts(self):
        print("📊 正在生成论文高级图表...")
        self._plot_boxplots()
        self._plot_radar_charts()
        self._plot_detail_timelines()
        print(f"✅ 所有图表已生成至 {self.plot_dir}/ 目录")

    def _plot_boxplots(self):
        """生成箱线图：展示100次运行的鲁棒性和方差分布"""
        metrics_to_plot = {
            'Spacing_Error': '平均间距跟踪误差 (m)\n↓ 越小越好',
            'TTC_Below_3s_Pct': 'TTC < 3s 危险时间占比 (%)\n↓ 越小越好',
            # 修复点 1：将 m/s³ 改为 m/s^3 避免字体库报错
            'Avg_Jerk': '加加速度均值 Jerk (m/s^3)\n↓ 越小越好 (舒适性)'
        }

        for metric, ylabel in metrics_to_plot.items():
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.df_metrics, x='Scenario', y=metric, hue='Algorithm',
                        palette='Set2', showfliers=False, width=0.7)

            plt.title(f'多场景不同算法鲁棒性对比 - {metric.replace("_", " ")}', fontsize=16, pad=15)
            plt.xlabel('测试场景 (直道 / 左弯 / 右弯)', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.legend(title='算法', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            plt.savefig(os.path.join(self.plot_dir, f"boxplot_{metric}.png"), dpi=300)
            plt.close()

    def _plot_radar_charts(self):
        """生成四维评价雷达图 (取倒数或归一化，使图形越大越优)"""
        labels = np.array(['安全性 (1/TTC风险)', '效率性 (速度误差倒数)', '舒适性 (1/Jerk)', '稳定性 (1/间距方差)'])
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # 按场景对数据求100次平均
        df_mean = self.df_metrics.groupby(['Scenario', 'Algorithm']).mean().reset_index()

        for scenario in self.scenarios:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            df_scenario = df_mean[df_mean['Scenario'] == scenario]

            for _, row in df_scenario.iterrows():
                # 指标映射逻辑 (将其转换为 0-100 的得分，分数越高越好)
                # 注：具体的映射基准值可根据你实际跑出的最大最小数据进行微调
                safety = max(0, 100 - row['TTC_Below_3s_Pct'] * 2)
                efficiency = max(0, 100 - row['Speed_Error'] * 15)
                comfort = max(0, 100 - row['Avg_Jerk'] * 30)
                stability = max(0, 100 - row['Spacing_Std'] * 20)

                values = [safety, efficiency, comfort, stability]
                values += values[:1]

                algo = row['Algorithm']
                linewidth = 3 if algo == 'MAPPO' else 1.5
                alpha = 0.2 if algo == 'MAPPO' else 0.05

                ax.plot(angles, values, linewidth=linewidth, label=algo)
                ax.fill(angles, values, alpha=alpha)

            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
            ax.set_ylim(0, 100)  # 统一雷达图比例尺
            plt.title(f'多算法综合性能雷达图 - {scenario.upper()}', size=16, y=1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

            plt.savefig(os.path.join(self.plot_dir, f"radar_{scenario}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_detail_timelines(self):
        """读取保存的 .npz 文件，绘制论文级别的时序微观响应图"""
        scenario = 'straight'
        colors = {'Traditional_CACC': '#95a5a6', 'MPC_CACC': '#f39c12',
                  'DQN': '#e74c3c', 'DDPG': '#3498db', 'MAPPO': '#2ecc71', 'PI-MAPPO': '#9b59b6'}  # A-MAPPO为尊贵的紫色

        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        for algo in self.algorithms:
            file_path = os.path.join(self.data_dir, f"{scenario}_{algo}_ep0.npz")
            if not os.path.exists(file_path):
                continue

            data = np.load(file_path, allow_pickle=True)
            time = data['time']
            # 取跟随车1 (index 0) 的数据作为代表
            speed = data['states'][:, 0, 0]

            # 修复点 2：直接从对齐了长度的 states 张量中提取车间距 (索引为2)，解决维度不匹配报错
            spacing = data['states'][:, 0, 2]

            accel = data['states'][:, 0, 1]

            ttc = []
            for state in data['states']:
                rel_v = state[0, 3]  # 前车相对自车的速度
                space = state[0, 2]
                val = space / abs(rel_v) if rel_v < -0.1 else 20.0
                ttc.append(min(20.0, val))

            lw = 2.5 if algo == 'MAPPO' else 1.5
            alpha = 1.0 if algo == 'MAPPO' else 0.7

            # 确保 time 和 spacing 长度强行对齐 (以防极个别情况差1位)
            min_len = min(len(time), len(speed), len(spacing), len(accel), len(ttc))

            axes[0].plot(time[:min_len], speed[:min_len], label=algo, color=colors[algo], linewidth=lw, alpha=alpha)
            axes[1].plot(time[:min_len], spacing[:min_len], label=algo, color=colors[algo], linewidth=lw, alpha=alpha)
            axes[2].plot(time[:min_len], accel[:min_len], label=algo, color=colors[algo], linewidth=lw, alpha=alpha)
            axes[3].plot(time[:min_len], ttc[:min_len], label=algo, color=colors[algo], linewidth=lw, alpha=alpha)

        # 补全领头车速度线
        try:
            lead_v = data['leader_speeds']
            axes[0].plot(time[:len(lead_v)], lead_v, 'k--', linewidth=2, label='领头车(Leader)')
        except:
            pass

        axes[0].set_ylabel('速度 (m/s)', fontsize=12)
        axes[0].set_title('车辆速度跟踪能力对比', fontsize=14)
        axes[0].legend(loc='lower right', ncol=3)

        axes[1].axhline(y=15.0, color='k', linestyle=':', label='期望间距 15m')
        axes[1].set_ylabel('间距 (m)', fontsize=12)
        axes[1].set_title('车间距保持能力对比 (误差越小越好)', fontsize=14)
        axes[1].legend(loc='upper right')

        axes[2].set_ylabel('加速度 (m/s²)', fontsize=12)
        axes[2].set_title('加速度平顺性对比 (衡量舒适度 Jerk)', fontsize=14)

        axes[3].axhline(y=3.0, color='r', linestyle='--', linewidth=2, label='安全临界线 (3s)')
        axes[3].set_ylabel('TTC (s)', fontsize=12)
        axes[3].set_ylim(0, 15)
        axes[3].set_xlabel('仿真时间 (s)', fontsize=12)
        axes[3].set_title('安全距离边界时间 (TTC 越大越安全)', fontsize=14)
        axes[3].legend(loc='upper right')

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "timeline_macro_details.png"), dpi=600)
        plt.close()



if __name__ == "__main__":
    print("🚀 启动 CACC 多维评估系统 (深度数据抽取版)...")
    evaluator = CACCEvaluator(episodes_per_scenario=100)
    evaluator.run_evaluation()