import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置Seaborn主题
sns.set_theme(style="darkgrid", palette="muted")

# 读取所有文件名开头为rewards_per_agent的文件
file_names = [file for file in os.listdir() if file.startswith('rewards_per_agent')]

for agent_number in range(1, 11):
    # 创建图形
    plt.figure(figsize=(12, 8))

    # 初始化一个空DataFrame来存储所有文件的Agent列的值
    all_rewards = pd.DataFrame()

    # 逐个读取文件，并将当前Agent列的值添加到DataFrame中
    for file_name in file_names:
        data = pd.read_csv(file_name)
        data.columns = [col.strip() for col in data.columns]
        agent_column = f'Agent {agent_number}'
        if agent_column in data.columns:
            all_rewards[file_name] = data[agent_column]
        else:
            print(f"Warning: {agent_column} not found in {file_name}")

    # 计算所有文件中当前Agent列的值的平均值
    average_reward = all_rewards.mean(axis=1)

    # 保存平均值到csv文件
    output_file_name = f'KD_NoFed_Agent{agent_number}.csv'
    average_reward.to_csv(output_file_name, index=False)

    # 绘制平均曲线
    sns.lineplot(x=range(len(average_reward)), y=average_reward, linewidth=3, color='#0077b6', label='Average Reward')

    # 添加图例、标题和坐标轴标签
    plt.legend(fontsize=16)
    plt.title(f'Average Reward for Agent {agent_number}', fontsize=20)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Average Reward', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # 保存图形
    plt.savefig(f'Average_Reward_Agent_{agent_number}.png')

    # 展示图形
    plt.show()
