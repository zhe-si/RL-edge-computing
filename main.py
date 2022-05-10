#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/10.

using tensorflow 2.3 with cuda 10.1
"""
import time
import gym
import numpy as np

from ddpg import DDPG, MEMORY_CAPACITY


#####################  hyper parameters  ####################
MAX_EPISODES = 200  # 最大训练轮次
MAX_EP_STEPS = 200  # 单轮训练最大步数

RENDER = True  # 展示动画
ENV_NAME = 'Pendulum-v0'  # gym环境名称


###############################  training  ####################################
def main():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = 3  # 探索程度参数，越大越探索
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        # 单轮训练累计奖励
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            a = ddpg.choose_action(s)
            # 以输出的a为中心，以var为标准差，生成一个符合正态分布的随机数(探索)，并限制其上下界
            a = np.clip(np.random.normal(a, var), -2, 2)
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            # 经验池满，开始训练
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995  # 逐步衰减探索程度
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                break
    print('Running time: ', time.time() - t1)


if __name__ == '__main__':
    main()
