import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traci

# 导入你的环境和算法
from cacc_real_time_env import CACCRealTimeEnv
from mappo import MAPPO
from pi_mappo import PI_MAPPO
from baseline_agents import DQN_Agent, DDPG_Agent, Traditional_CACC, MPC_CACC

# ==========================================
# 全局学术级图表格式设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
sns.set_theme(style="ticks", font=['SimHei', 'Arial Unicode MS'], rc={'axes.unicode_minus': False})


def generate_string_stability_plots():
    print("🚀 启动串稳定性微观测试，准备生成论文三联图...")

    out_dir = "evaluation_results/plots/string_stability"
    os.makedirs(out_dir, exist_ok=True)

    algorithms = ['Traditional_CACC', 'MPC_CACC', 'DQN', 'DDPG', 'MAPPO', 'PI-MAPPO']

    # 实例化算法
    agents = {
        'Traditional_CACC': Traditional_CACC(),
        'MPC_CACC': MPC_CACC(),
        'DQN': DQN_Agent(3, 9),
        'DDPG': DDPG_Agent(3, 9),
        'MAPPO': MAPPO(3, 9, 1),
        'PI-MAPPO': PI_MAPPO(3, 9, 1)
    }

    # 加载已训练的最佳模型权重
    for algo in ['DQN', 'DDPG', 'MAPPO']:
        model_path = f'models/best_{algo.lower()}.pth'
        if os.path.exists(model_path):
            agents[algo].load_models(model_path)

    if os.path.exists('models/best_pimappo.pth'):
        agents['PI-MAPPO'].load_models('models/best_pimappo.pth')

    # 配置环境（直道，延长步数以便观察扰动的传播与衰减）
    env_config = {
        'scenario_type': 'straight',
        'gui': False,
        'max_steps_per_episode': 800  # 约80秒仿真，足够观察前车扰动传播
    }
    env = CACCRealTimeEnv(env_config)

    # 车辆配色：领头车(黑色虚线), 跟随车1(红色), 跟随车2(蓝色), 跟随车3(绿色)
    colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71']
    labels = ['领头车 (Leader)', '跟随车1 (Follower 1)', '跟随车2 (Follower 2)', '跟随车3 (Follower 3)']
    line_styles = ['--', '-', '-', '-']

    for algo in algorithms:
        print(f"🚗 正在运行测试并绘制算法: {algo}")
        states = env.reset()
        agent = agents[algo]

        # 数据缓存字典
        step_data = {veh: {'pos': [], 'vel': [], 'acc': []} for veh in env.all_vehicles}
        times = []

        done = False
        step = 0

        # 执行仿真并抓取底层数据
        while not done and step < env_config['max_steps_per_episode']:
            # 适配不同算法的动作选择接口
            if hasattr(agent, 'select_action'):
                actions = agent.select_action(states, exploration=False)
            else:
                actions = agent.get_actions(states, exploration=False)

            states, _, done, _ = env.step(actions)

            times.append(env.current_time)

            # 使用 SUMO Traci 接口提取每一辆车的绝对信息
            for veh in env.all_vehicles:
                try:
                    p = traci.vehicle.getLanePosition(veh)
                    v = traci.vehicle.getSpeed(veh)
                    a = traci.vehicle.getAcceleration(veh)
                except:
                    p, v, a = 0.0, 0.0, 0.0
                step_data[veh]['pos'].append(p)
                step_data[veh]['vel'].append(v)
                step_data[veh]['acc'].append(a)

            step += 1

        # ==========================================
        # 绘图逻辑 (生成 3x1 的论文规格子图)
        # ==========================================
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        for idx, veh in enumerate(env.all_vehicles):
            # 位置图
            axes[0].plot(times, step_data[veh]['pos'], color=colors[idx], linestyle=line_styles[idx],
                         label=labels[idx], linewidth=2.0)
            # 速度图
            axes[1].plot(times, step_data[veh]['vel'], color=colors[idx], linestyle=line_styles[idx],
                         label=labels[idx], linewidth=2.0)
            # 加速度图
            axes[2].plot(times, step_data[veh]['acc'], color=colors[idx], linestyle=line_styles[idx],
                         label=labels[idx], linewidth=2.0)

        # 1. 格式化位置图 (Position)
        axes[0].set_ylabel('位置 Position (m)', fontsize=13, fontweight='bold')
        axes[0].set_title(f'[{algo}] 车辆位置轨迹 (Position Trajectory)', fontsize=15)
        axes[0].legend(loc='upper left', fontsize=11)
        axes[0].grid(axis='y', linestyle='--', alpha=0.6)

        # 2. 格式化速度图 (Velocity)
        axes[1].set_ylabel('速度 Velocity (m/s)', fontsize=13, fontweight='bold')
        axes[1].set_title(f'[{algo}] 速度扰动传播 (Velocity Disturbance Propagation)', fontsize=15)
        axes[1].grid(axis='y', linestyle='--', alpha=0.6)

        # 3. 格式化加速度图 (Acceleration)
        axes[2].set_ylabel('加速度 Accel (m/s²)', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('仿真时间 Time (s)', fontsize=13, fontweight='bold')
        axes[2].set_title(f'[{algo}] 加速度扰动传播 (Acceleration Disturbance Propagation)', fontsize=15)
        axes[2].grid(axis='y', linestyle='--', alpha=0.6)

        sns.despine()  # 移除上边框和右边框，提高学术感
        plt.tight_layout()

        # 保存高清大图
        save_path = os.path.join(out_dir, f"StringStability_{algo}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    env.close()
    print(f"\n✅ 所有算法的串稳定性三联图已成功保存至目录：{out_dir}")


if __name__ == "__main__":
    generate_string_stability_plots()