"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import time
import gym
import numpy as np
import tensorflow as tf2

from ddpg import DDPG, MEMORY_CAPACITY

tf = tf2.compat.v1
tf.disable_v2_behavior()


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

    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                break
    print('Running time: ', time.time() - t1)


if __name__ == '__main__':
    main()
