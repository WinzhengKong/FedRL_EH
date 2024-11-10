import torch
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
from policy import PolicyNetwork

device = torch.device("cpu")  # 使用CPU进行评估

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

# 绘制评估曲线
def plot_eval_results(eval_rewards, model_save_path):
    plt.plot(eval_rewards, label='Average Reward per Model')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Model Evaluation Across Episodes')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{model_save_path}/evaluation_results.png")
    plt.show()

# 主函数，读取模型并进行评估
def main(model_save_path, num_episodes, num_seeds=100):
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
        model_files = [f for f in os.listdir(model_save_path) if f.startswith(model_filename_prefix) and f.endswith('.pth')]

        if model_files:
            model_path = os.path.join(model_save_path, model_files[0])  # 选择第一个匹配的模型文件
            print(f"Loading model from {model_path}")
            global_model = load_model(model_path, global_model)

            # 评估当前模型的表现
            rewards_across_seeds = [evaluate_model(env, global_model, num_episodes=1, seed=s) for s in range(1, num_seeds + 1)]
            mean_reward = np.mean(rewards_across_seeds)
            eval_rewards.append(mean_reward)
            print(f"Model {model_path} has been evaluated")
        else:
            print(f"Model file with prefix {model_filename_prefix} not found, skipping.")

    # 绘制评估结果
    plot_eval_results(eval_rewards, model_save_path)

# 运行评估
if __name__ == "__main__":
    model_save_path = 'E:/paperbymyself/1021byzantine/log/model_20241110_151531'  # 设置您的模型存储路径
    num_episodes = 2000  # 评估模型的总轮数
    main(model_save_path, num_episodes)
