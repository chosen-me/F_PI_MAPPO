import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cacc_real_time_env import CACCRealTimeEnv
from mappo import MAPPO
from pi_mappo import PI_MAPPO
from baseline_agents import DQN_Agent, DDPG_Agent, Traditional_CACC, MPC_CACC

# ==========================================
# 全局学术级图表格式设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font=['SimHei', 'Arial Unicode MS'], rc={'axes.unicode_minus': False})


def generate_individual_timeseries_charts():
    print("🚗 正在运行 1 回合仿真以抓取实时微观数据...")

    # 包含所有需要对比的6种算法
    algorithms = ['Traditional_CACC', 'MPC_CACC', 'DQN', 'DDPG', 'MAPPO', 'PI-MAPPO']

    # 【修复点1】：将这里的跟随车数量全部从 3 改为 5，以匹配训练好的模型维度 (45维)
    agents = {
        'Traditional_CACC': Traditional_CACC(),
        'MPC_CACC': MPC_CACC(num_agents=5),
        'DQN': DQN_Agent(5, 9),
        'DDPG': DDPG_Agent(5, 9),
        'MAPPO': MAPPO(5, 9, 1),
        'PI-MAPPO': PI_MAPPO(5, 9, 1)
    }

    # 加载预训练模型权重
    for algo in ['DQN', 'DDPG', 'MAPPO', 'PI-MAPPO']:
        model_path = f'models/best_{algo.replace("-", "").lower()}.pth'
        if os.path.exists(model_path):
            agents[algo].load_models(model_path)
            print(f"✅ 成功加载 {algo} 模型权重！")
        else:
            print(f"⚠️ 未找到 {algo} 模型权重: {model_path}")

    # 截取前 400 步 (40秒) 作为分析窗口，此时动态变化最剧烈
    env_config = {'scenario_type': 'straight', 'gui': False, 'max_steps_per_episode': 400}
    # 【修复点2】：将环境中的跟随车辆数量设为 5
    env = CACCRealTimeEnv(env_config, num_followers=5)

    all_step_data = []

    for algo in algorithms:
        print(f"  ▶ 正在抓取算法运行数据: {algo}")
        states = env.reset()
        agent = agents[algo]
        for step in range(env_config['max_steps_per_episode']):
            actions = agent.select_action(states, exploration=False) if hasattr(agent,
                                                                                'select_action') else agent.get_actions(
                states, exploration=False)
            states, _, done, info = env.step(actions)

            # 记录跟驰车 1 的实时状态
            v_ego = states[0][0]
            spacing = states[0][2]
            rel_speed = states[0][3]
            accel = states[0][1]
            ttc = spacing / abs(rel_speed) if rel_speed < -0.1 else 20.0

            all_step_data.append({
                'Time (s)': step * 0.1,
                'Algorithm': algo,
                'Speed (m/s)': v_ego,  # 提取自车速度
                'Leader Speed (m/s)': info['leader_speed'],  # 提取领头车速度
                'Spacing (m)': spacing,
                'TTC (s)': min(ttc, 20.0),
                'Acceleration (m/s²)': accel
            })
            if done: break
    env.close()

    df_steps = pd.DataFrame(all_step_data)

    out_dir = os.path.join("final", "plots")
    os.makedirs(out_dir, exist_ok=True)

    # 【修复点3】：更换为高辨识度的清新莫兰迪/马卡龙配色，杜绝颜色重复
    my_palette = {
        'Traditional_CACC': '#A0AAB2',  # 清新灰 (Soft Slate)
        'MPC_CACC': '#FFCA3A',          # 明亮黄 (Sunny Yellow)
        'DQN': '#FFB7B2',               # 蜜桃粉 (Peach Blush)
        'DDPG': '#1982C4',              # 湖水蓝 (Cerulean Blue)
        'MAPPO': '#8AC926',             # 薄荷绿 (Mint Green)
        'PI-MAPPO': '#00C7B1'           # 活力青 (Elegant Purple)
    }

    # 通用的图表美化参数
    fig_size_wide = (12, 4.5)
    fig_size = (10, 5)
    line_width = 2.0
    title_font = 16
    label_font = 13
    tick_font = 12

    # ==========================================
    # 图 1：实时速度跟踪 (Speed)
    # ==========================================
    plt.figure(figsize=fig_size_wide)

    # 画出所有算法的速度曲线
    sns.lineplot(data=df_steps, x='Time (s)', y='Speed (m/s)', hue='Algorithm',
                 palette=my_palette, linewidth=line_width, dashes=False)

    # 领头车换成高级的深炭灰色，避免和纯黑或算法颜色混淆
    leader_data = df_steps[df_steps['Algorithm'] == 'PI-MAPPO']
    plt.plot(leader_data['Time (s)'], leader_data['Leader Speed (m/s)'],
             color='#2C3E50', linestyle='--', linewidth=2.5, label='领头车 (Leader)')

    plt.title('车辆速度跟踪能力对比', fontsize=title_font, fontweight='bold', pad=15)
    plt.xlabel('仿真时间 (秒)', fontsize=label_font)
    plt.ylabel('速度 (m/s)', fontsize=label_font)
    plt.xticks(fontsize=tick_font)
    plt.yticks(fontsize=tick_font)

    plt.legend(loc='lower right', fontsize=11, ncol=3)

    save_path_0 = os.path.join(out_dir, "micro_speed_academic.png")
    plt.savefig(save_path_0, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 图 2：实时车间距跟踪 (Spacing)
    # ==========================================
    plt.figure(figsize=fig_size)
    ax1 = sns.lineplot(data=df_steps, x='Time (s)', y='Spacing (m)', hue='Algorithm',
                       palette=my_palette, linewidth=line_width, dashes=False)
    plt.axhline(15.0, color='#E74C3C', linestyle='--', linewidth=2.0, label='目标间距 (15m)')

    plt.title('协同编队微观时序动态：实时车间距跟踪', fontsize=title_font, fontweight='bold', pad=15)
    plt.xlabel('仿真时间 (秒)', fontsize=label_font)
    plt.ylabel('车间距 (m)', fontsize=label_font)
    plt.xticks(fontsize=tick_font)
    plt.yticks(fontsize=tick_font)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, title='对比算法', title_fontsize=13)

    save_path_1 = os.path.join(out_dir, "micro_spacing_academic.png")
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 图 3：实时 TTC 安全裕度 (TTC)
    # ==========================================
    plt.figure(figsize=fig_size)
    ax2 = sns.lineplot(data=df_steps, x='Time (s)', y='TTC (s)', hue='Algorithm',
                       palette=my_palette, linewidth=line_width, dashes=False)
    plt.axhline(2.5, color='#E74C3C', linestyle='--', linewidth=2.0, label='安全红线 (TTC=2.5s)')

    plt.title('协同编队微观时序动态：碰撞时间余量 (TTC)', fontsize=title_font, fontweight='bold', pad=15)
    plt.xlabel('仿真时间 (秒)', fontsize=label_font)
    plt.ylabel('TTC (s)', fontsize=label_font)
    plt.xticks(fontsize=tick_font)
    plt.yticks(fontsize=tick_font)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, title='对比算法', title_fontsize=13)

    save_path_2 = os.path.join(out_dir, "micro_ttc_academic.png")
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 图 4：实时加速度波动 (Acceleration / Comfort)
    # ==========================================
    plt.figure(figsize=fig_size)
    ax3 = sns.lineplot(data=df_steps, x='Time (s)', y='Acceleration (m/s²)', hue='Algorithm',
                       palette=my_palette, linewidth=line_width, dashes=False)

    plt.title('协同编队微观时序动态：实时加速度波动', fontsize=title_font, fontweight='bold', pad=15)
    plt.xlabel('仿真时间 (秒)', fontsize=label_font)
    plt.ylabel('加速度 (m/s²)', fontsize=label_font)
    plt.xticks(fontsize=tick_font)
    plt.yticks(fontsize=tick_font)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, title='对比算法', title_fontsize=13)

    save_path_3 = os.path.join(out_dir, "micro_accel_academic.png")
    plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 四张独立的高清微观时序图已生成至: {out_dir}")


if __name__ == "__main__":
    generate_individual_timeseries_charts()
