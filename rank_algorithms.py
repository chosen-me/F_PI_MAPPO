import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def calculate_scores_and_rank():
    # 1. 读取数据
    data_path = "evaluation_results_off/data/comprehensive_metrics_report.csv"
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件 {data_path}，请先运行评估脚本。")
        return

    df = pd.read_csv(data_path)
    df_mean = df.groupby('Algorithm').mean(numeric_only=True).reset_index()

    # 2. 定义评价维度与对应的负向指标 (越小越好)
    dimensions = {
        'Stability': ['Spacing_Error', 'Spacing_Std'],
        'Comfort': ['Avg_Jerk', 'Accel_RMS'],
        'Efficiency': ['Speed_Error']
    }

    # 维度权重设置
    weights = {
        'Safety': 0.40,
        'Stability': 0.30,
        'Comfort': 0.15,
        'Efficiency': 0.15
    }

    # 3. 极值归一化函数 (0-100分制)
    def normalize_to_score(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(100.0, index=series.index)
        return (max_val - series) / (max_val - min_val) * 100.0

    # 4. 计算各维度单项得分
    scores_df = pd.DataFrame({'Algorithm': df_mean['Algorithm']})

    # 【核心修复】：安全性采用绝对扣分制，与雷达图计算逻辑完全统一
    # 基础分 100，TTC<3s 每增加 1% 扣 2 分，每次碰撞扣 20 分
    safety_score = 100.0 - df_mean['TTC_Below_3s_Pct'] * 2.0 - df_mean['Collision_Count'] * 20.0
    scores_df['Safety'] = np.clip(safety_score, 0, 100)

    # 其他维度依然使用相对极值归一化
    for dim, metrics in dimensions.items():
        dim_score = np.zeros(len(df_mean))
        for m in metrics:
            dim_score += normalize_to_score(df_mean[m])
        scores_df[dim] = dim_score / len(metrics)

    # 5. 计算综合加权总分
    scores_df['Total_Score'] = (
            scores_df['Safety'] * weights['Safety'] +
            scores_df['Stability'] * weights['Stability'] +
            scores_df['Comfort'] * weights['Comfort'] +
            scores_df['Efficiency'] * weights['Efficiency']
    )

    # 按照总分降序排列
    scores_df = scores_df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)

    # ==========================================
    # 打印精美排行榜
    # ==========================================
    print("\n" + "🏆" * 25)
    print("      CACC 算法综合性能权威排行榜      ")
    print("🏆" * 25)
    print(
        f"{'排名':<4} | {'算法名称':<18} | {'综合总分':<8} | {'安全性(40%)':<10} | {'稳定性(30%)':<10} | {'舒适性(15%)':<10} | {'效率性(15%)':<10}")
    print("-" * 85)

    for i, row in scores_df.iterrows():
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f" {i + 1} "
        print(
            f" {medal:<3} | {row['Algorithm']:<16} | {row['Total_Score']:>6.2f}分 | {row['Safety']:>8.2f}分 | {row['Stability']:>8.2f}分 | {row['Comfort']:>8.2f}分 | {row['Efficiency']:>8.2f}分")
    print("-" * 85)

    plot_stacked_bar(scores_df, weights)


def plot_stacked_bar(scores_df, weights):
    plot_df = scores_df.copy()
    plot_df['Safety_Contribution'] = plot_df['Safety'] * weights['Safety']
    plot_df['Stability_Contribution'] = plot_df['Stability'] * weights['Stability']
    plot_df['Comfort_Contribution'] = plot_df['Comfort'] * weights['Comfort']
    plot_df['Efficiency_Contribution'] = plot_df['Efficiency'] * weights['Efficiency']

    algorithms = plot_df['Algorithm'].tolist()
    safety_vals = plot_df['Safety_Contribution'].values
    stability_vals = plot_df['Stability_Contribution'].values
    comfort_vals = plot_df['Comfort_Contribution'].values
    efficiency_vals = plot_df['Efficiency_Contribution'].values

    fig, ax = plt.subplots(figsize=(12, 7))

    p1 = ax.bar(algorithms, safety_vals, label='安全性贡献 (40%)', color='#e74c3c', width=0.5)
    p2 = ax.bar(algorithms, stability_vals, bottom=safety_vals, label='稳定性贡献 (30%)', color='#3498db', width=0.5)
    p3 = ax.bar(algorithms, comfort_vals, bottom=safety_vals + stability_vals, label='舒适性贡献 (15%)',
                color='#2ecc71', width=0.5)
    p4 = ax.bar(algorithms, efficiency_vals, bottom=safety_vals + stability_vals + comfort_vals,
                label='效率性贡献 (15%)', color='#f1c40f', width=0.5)

    for i, total in enumerate(plot_df['Total_Score']):
        ax.text(i, total + 1, f"{total:.1f}分", ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('综合性能得分 (0-100)', fontsize=14)
    ax.set_title('多算法综合性能得分构成对比', fontsize=16, pad=15)
    plt.xticks(rotation=15, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_dir = "evaluation_results_off/plots"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "algorithm_ranking_score.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 算法得分堆叠图已生成：{save_path}")


if __name__ == "__main__":
    calculate_scores_and_rank()