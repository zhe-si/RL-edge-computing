#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/10.

using tensorflow 2.3 with cuda 10.1, python 3.8
"""
import sys

import numpy as np

from models.ddpg import DDPG, MEMORY_CAPACITY
from envs import AbsEnv, GymEnv
from tools import show_run_time

#####################  hyper parameters  ####################
IS_TRAIN_MODE = True  # 训练、验证模式

MAX_EPISODES = 200  # 最大训练轮次
MAX_EP_STEPS = 200  # 单轮训练最大步数


###############################  training  ####################################
@show_run_time()
def train(ddpg, env: AbsEnv):
    """训练"""
    max_reward = -float('inf')
    explore_degree = 3  # 探索程度参数，越大越探索
    all_step = 0
    for i in range(MAX_EPISODES):
        s = env.reset()
        # 单轮训练累计奖励
        ep_reward = 0.0
        for j in range(MAX_EP_STEPS):
            a = ddpg.choose_action(s)
            # 以输出的a为中心，以var为标准差，生成一个符合正态分布的随机数(探索)
            a = np.random.normal(a, explore_degree)
            # 限制动作范围，并重映射动作值到正确范围
            a = env.standard_action(a)
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            # 经验池满，开始训练
            if ddpg.pointer > MEMORY_CAPACITY:
                explore_degree *= .9995  # 逐步衰减探索程度
                ddpg.learn()
                ddpg.print_tensorboard({'loss': ddpg.cal_loss(), 'reward': r}, all_step)

            s = s_
            ep_reward += r
            all_step += 1

            if j == MAX_EP_STEPS - 1 or done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % explore_degree, )
                if i % 5 == 0 and ep_reward > max_reward:
                    ddpg.save_model(global_step=i)
                    max_reward = ep_reward
                break


###############################  experiment  ####################################
def experiment(ddpg, env: AbsEnv, is_reboot=False):
    """评估、验证、使用模型"""
    while True:
        s = env.reset()
        print('start new experiment')

        ep_reward = 0.0
        step_num = 0

        while True:
            a = ddpg.choose_action(s)
            # 限制动作范围，并重映射动作值到正确范围
            a = env.standard_action(a)
            s, r, done, info = env.step(a)

            ep_reward += r
            step_num += 1
            print(f'Choose action: {a}, Reward avg: {ep_reward / step_num}')

            if done:
                break
        if not is_reboot:
            break


def main():
    env = GymEnv('Pendulum-v0', is_render=True)

    ddpg = DDPG(env.action_dim(), env.observation_dim(), env.action_radius())
    is_load = ddpg.load_model()

    if IS_TRAIN_MODE:
        train(ddpg, env)
    else:
        if not is_load:
            print('WARNING: 模型验证模式下，模型加载失败', file=sys.stderr)
        experiment(ddpg, env, True)


if __name__ == '__main__':
    main()
