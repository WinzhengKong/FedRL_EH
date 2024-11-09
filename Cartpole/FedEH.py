import gym
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
from policy import PolicyNetwork
from custom_cartpole import make_custom_cartpole_env
import torch.nn.functional as F

# 创建全局模型
global_model = None
device = torch.device("cpu")

# 定义保存模型函数
def save_model(model, path, episode):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(path):
        os.makedirs(path)
    model_filename = f"global_model_episode_{episode}_{timestamp}.pth"
    model_path = os.path.join(path, model_filename)
    torch.save(model.state_dict(), model_path)


class Agent:
    def __init__(self, env, state_size, action_size, hidden_sizes, lr, activation_fn, gamma, global_model, device):
        self.env = env
        self.gamma = gamma
        self.device = device
        # 初始化智能体的模型为全局模型的副本
        self.policy = PolicyNetwork(state_size, action_size, hidden_sizes, activation_fn).to(self.device)
        self.policy.load_state_dict(global_model.state_dict())  # 同步全局模型的参数
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

    def train_reinforce(self):
        state = torch.FloatTensor(self.env.reset()).unsqueeze(0).to(device)
        log_probs = []
        rewards = []
        done = False
        while not done:
            probs = self.policy(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            next_state, reward, done, _ = self.env.step(action.item())
            state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # baseline

        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return sum(rewards), self.policy.state_dict()


def fed_avg_params(global_model, local_models, num_agents):
    # 初始化全局模型参数为0
    new_params = {key: torch.zeros_like(param) for key, param in global_model.state_dict().items()}

    # 将所有智能体的模型参数进行累加
    for local_model in local_models:
        for key, param in local_model.items():
            new_params[key] += param

    # 计算平均值
    for key in new_params:
        new_params[key] /= num_agents

    # 更新全局模型参数
    global_model.load_state_dict(new_params)

def evaluate_model(env, model, num_episodes, seed):
    total_rewards = []
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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

def main(seed, episodes, model_save_path):
    global global_model
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    agents_count = 5

    # 定义每个智能体的环境异构，改变杆子的长度
    pole_lengths = [0.5, 0.5, 0.5, 0.5, 0.5]  # 每个智能体的杆子长度不同
    envs = [make_custom_cartpole_env(pole_lengths[i], max_steps=500) for i in range(agents_count)]  # 使用自定义环境创建实例

    state_size = envs[0].observation_space.shape[0]
    action_size = envs[0].action_space.n
    hidden_sizes_list = [128]
    activation_fn = 'relu'
    gamma = 0.99

    # 创建初始全局模型
    global_model = PolicyNetwork(state_size, action_size, hidden_sizes_list, activation_fn).to(device)
    global_optimizer = optim.Adam(global_model.parameters(), lr=6e-4)
    global_model.optimizer = global_optimizer

    agents = [Agent(envs[i], state_size, action_size, hidden_sizes_list, 6e-4, activation_fn, gamma, global_model, device=device) for i in range(agents_count)]

    rewards_per_agent = [[] for _ in range(agents_count)]
    average_rewards = []
    average_rewards_eval = []

    for episode in range(episodes):
        local_models = []
        total_rewards = []
        for agent in agents:
            reward, local_model = agent.train_reinforce()
            total_rewards.append(reward)
            local_models.append(local_model)

        for i, reward in enumerate(total_rewards):
            rewards_per_agent[i].append(reward)

        average_reward = sum(total_rewards) / len(total_rewards)
        average_rewards.append(average_reward)
        print(f"Episode {episode + 1}:")
        for idx, reward in enumerate(total_rewards):
            if len(rewards_per_agent[idx]) >= 100:
                last_100_avg = sum(rewards_per_agent[idx][-100:]) / 100
            else:
                last_100_avg = sum(rewards_per_agent[idx]) / len(rewards_per_agent[idx])
            print(f"  Agent {idx + 1} Reward: {reward} | Average Reward: {last_100_avg:.4f}")

        print(f"  Average Reward: {average_reward}")


        # 使用FedAvg对模型参数进行聚合
        fed_avg_params(global_model, local_models, agents_count)

        # 保存当前轮次的全局模型
        save_model(global_model, model_save_path, episode)

        if (episode + 1) % 10 == 0:
            # 在不同种子下评估当前轮次的全局模型
            env = gym.make('CartPole-v1')
            seeds = range(1, 101)  # 测试种子
            rewards_across_seeds = [evaluate_model(env, global_model, num_episodes=1, seed=s) for s in seeds]

            # 记录和输出平均奖励
            mean_reward = np.mean(rewards_across_seeds)
            average_rewards_eval.append(mean_reward)
            print(f"Episode {episode + 1}: Average Reward across seeds: {mean_reward}")



if __name__ == "__main__":
    seed = 42
    episodes = 2000
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = f'E:/paperbymyself/1021byzantine/log/model_{timestamp}'
    main(seed, episodes, model_save_path)
