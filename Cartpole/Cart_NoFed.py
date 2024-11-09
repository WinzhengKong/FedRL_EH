import gym
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from datetime import datetime
from torch.distributions import Categorical
from policy import PolicyNetwork
import random
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cpu")


class Agent:
    def __init__(self, env, state_size, action_size, hidden_sizes, lr, activation_fn, gamma, device):
        self.env = env
        self.gamma = gamma
        self.device = device
        self.policy = PolicyNetwork(state_size, action_size, hidden_sizes, activation_fn).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.activation_fn = activation_fn

    def train_reinforce(self):
        state = torch.FloatTensor(self.env.reset()).unsqueeze(0).to(device)
        log_probs = []
        rewards = []
        done = False

        while not done:
            probs = self.policy(state)
            dist = Categorical(probs)
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

        return sum(rewards)



def main(seed, episodes):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    agents_count = 5
    envs = [gym.make('CartPole-v1') for _ in range(agents_count)]
    pole_lengths = [0.5, 0.6, 0.7, 0.4, 0.3]  # 每个智能体的杆子长度不同
    for i, env in enumerate(envs):
        env.env.length = pole_lengths[i]  # 修改环境中的杆子长度
        env.seed(seed)

    state_size = envs[0].observation_space.shape[0]
    action_size = envs[0].action_space.n
    lr_list = [6e-4, 6e-4, 6e-4, 6e-4, 6e-4]
    activaton_list = ['relu', 'relu', 'relu', 'relu', 'relu']
    gamma = 0.99

    hidden_sizes_list = [[128], [128], [128], [128], [128]]
    agents = [Agent(envs[i], state_size, action_size, hidden_sizes_list[i % len(hidden_sizes_list)],
                    lr_list[i % len(lr_list)], activaton_list[i % len(hidden_sizes_list)], gamma, device=device) for i
              in range(agents_count)]


    rewards_per_agent = [[] for _ in range(agents_count)]
    average_rewards = []

    for episode in range(episodes):
        total_rewards = []

        for agent in agents:
            reward = agent.train_reinforce()
            total_rewards.append(reward)

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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    rewards_filename = f'rewards_per_agent_Cartpole_{timestamp}.csv'
    rewards_df = pd.DataFrame({f'Agent {i + 1}': rewards_per_agent[i] for i in range(agents_count)})
    rewards_df.to_csv(rewards_filename, index=False)



if __name__ == "__main__":
        seed = 42
        episodes = 2000
        print(f"Seed: {seed}")
        print(f"Episodes: {episodes}")
        main(seed, episodes)
