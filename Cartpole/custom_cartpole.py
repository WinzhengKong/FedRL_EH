import gym
from gym.envs.classic_control import CartPoleEnv
import numpy as np


class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, pole_length=0.5, max_steps=500):
        super(CustomCartPoleEnv, self).__init__()
        self.length = pole_length  # 设置杆子的长度
        self.max_steps = max_steps  # 设置最大步数
        self.step_count = 0  # 计数当前步数

    def reset(self):
        self.step_count = 0  # 重置步数计数
        return super(CustomCartPoleEnv, self).reset()

    def step(self, action):
        self.step_count += 1
        state, reward, done, info = super(CustomCartPoleEnv, self).step(action)

        # 当步数达到最大限制时，结束环境
        if self.step_count >= self.max_steps:
            done = True

        return state, reward, done, info


# 注册不同杆子长度的环境
def make_custom_cartpole_env(pole_length, max_steps=500):
    return CustomCartPoleEnv(pole_length=pole_length, max_steps=max_steps)


# 测试自定义环境
if __name__ == '__main__':
    env = make_custom_cartpole_env(pole_length=0.1)  # 这里可以调整杆子长度
    state = env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if done:
            state = env.reset()
    env.close()
