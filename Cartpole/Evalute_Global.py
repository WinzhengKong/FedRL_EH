import torch
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
from policy import PolicyNetwork

device = torch.device("cpu")  # 使用CPU进行评估


# 平滑曲线的函数
def smooth_curve(values, window_size=20):
    """
    使用滑动平均对给定数据进行平滑
    :param values: 要平滑的数据
    :param window_size: 滑动窗口大小
    :return: 平滑后的数据
    """
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

# 绘制评估曲线
def plot_eval_results(eval_rewards, model_save_path, window_size=20):
    # 对评估结果进行平滑
    smoothed_rewards = smooth_curve(eval_rewards, window_size)

    # 绘制平滑后的曲线
    plt.plot(smoothed_rewards, label='Smoothed Average Reward per Model')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Model Evaluation Across Episodes (Smoothed)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{model_save_path}/evaluation_results_smoothed.png")
    plt.show()

# 评估模型的函数（与之前代码中的 evaluate_model 一样）
def evaluate_model(env, model, num_episodes, seed):
    total_rewards = []
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    for _ in range(num_episodes):
        state = torch.FloatTensor(env.reset()).unsqueeze(0).to(device)
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                probs = model(state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# 读取模型文件并加载
def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换模型为评估模式
    return model

# 主函数，读取模型并进行评估
def main(model_save_path, num_episodes, num_seeds=10):
    # 创建一个Gym环境用于评估
    env = gym.make('CartPole-v1')

    # 初始化模型和记录评估结果的列表
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_sizes_list = [128]
    activation_fn = 'relu'
    global_model = PolicyNetwork(state_size, action_size, hidden_sizes_list, activation_fn).to(device)

    eval_rewards = []

    # 逐个加载每个模型并进行评估
    for episode in range(num_episodes):
        # 生成匹配模型文件的前缀（忽略时间戳部分）
        model_filename_prefix = f"global_model_episode_{episode}_"
        model_files = [f for f in os.listdir(model_save_path)
                       if f.startswith(model_filename_prefix) and f.endswith('.pth')]

        if model_files:
            model_path = os.path.join(model_save_path, model_files[0])  # 选择第一个匹配的模型文件
            print(f"Loading model from {model_path}")
            global_model = load_model(model_path, global_model)

            # 评估当前模型的表现
            rewards_across_seeds = [evaluate_model(env, global_model, num_episodes=1, seed=s)
                                    for s in range(1, num_seeds+1)]
            mean_reward = np.mean(rewards_across_seeds)
            eval_rewards.append(mean_reward)
            print(f"Model {model_path} has been evaluated")
        else:
            print(f"Model file with prefix {model_filename_prefix} not found, skipping.")

    # 绘制评估结果
    plot_eval_results(eval_rewards, model_save_path)

# 运行评估
if __name__ == "__main__":
    model_save_path = 'E:/paperbymyself/1021byzantine/log/model_20241110_212957'  # 设置您的模型存储路径
    num_episodes = 2000  # 评估模型的总轮数
    main(model_save_path, num_episodes)
