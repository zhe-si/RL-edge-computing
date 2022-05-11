#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/11.
"""
import gym
import numpy as np

from envs.abs_env import AbsEnv


class GymEnv(AbsEnv):
    def __init__(self, env_name, is_render=False):
        self.env = gym.make(env_name)
        self.env = self.env.unwrapped
        self.env.seed(1)
        self.is_render = is_render

    def observation_dim(self) -> int:
        return self.env.observation_space.shape[0]

    def action_dim(self) -> int:
        return self.env.action_space.shape[0]

    def is_discrete(self) -> bool:
        return False

    def standard_action(self, action):
        a_high = self.action_high()
        a_low = self.action_low()
        action = [np.clip(action[i], a_low[i], a_high[i]) for i in range(len(action))]
        return np.array(action)

    def action_radius(self):
        return self.action_high()

    def action_high(self):
        return self.env.action_space.high

    def action_low(self):
        return self.env.action_space.low

    def reset(self):
        return self.env.reset()

    def step(self, action):
        result = self.env.step(action)
        if self.is_render:
            self.env.render()
        return result
