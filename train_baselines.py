import numpy as np
import os
from tqdm import tqdm
from cacc_real_time_env import CACCRealTimeEnv
from baseline_agents import DQN_Agent, DDPG_Agent


def train_baseline(algorithm_name="DDPG", num_episodes=500):
    print(f"🚀 开始训练基线算法: {algorithm_name}")

    # 基础配置
    config = {
        'scenario_type': 'straight',
        'gui': False,
        'max_steps_per_episode': 500
    }
    env = CACCRealTimeEnv(config)

    # 初始化智能体
    if algorithm_name == "DQN":
        agent = DQN_Agent(num_agents=3, state_dim=9)
    else:
        agent = DDPG_Agent(num_agents=3, state_dim=9)

    os.makedirs('models', exist_ok=True)
    best_reward = -float('inf')

    for episode in range(1, num_episodes + 1):
        states = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 500:
            actions = agent.select_action(states, exploration=True)
            next_states, rewards, done, _ = env.step(actions)

            # 将每个跟驰车的经验存入回放池
            for i in range(3):
                agent.memory.push(states[i], actions[i], rewards[i], next_states[i], done)

            # 每步都进行网络更新 (DDPG和DQN是Off-policy算法)
            agent.update()

            states = next_states
            episode_reward += sum(rewards)
            step += 1

        print(f"回合 {episode:3d} | 总奖励: {episode_reward:.2f}")

        # 保存表现最好的模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_models(f'models/best_{algorithm_name.lower()}.pth')
            print(f"  🌟 发现新最佳模型! 保存至 models/best_{algorithm_name.lower()}.pth")

    env.close()
    print(f"🏁 {algorithm_name} 训练完成！")


if __name__ == "__main__":
    # 分别训练 DQN 和 DDPG
    # 如果嫌时间长，跑 300 到 500 个回合即可提取一个足够用于对比的模型
    train_baseline("DQN", num_episodes=400)
    train_baseline("DDPG", num_episodes=400)