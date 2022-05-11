#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/11.
"""
from abc import ABCMeta, abstractmethod


class AbsEnv(metaclass=ABCMeta):
    """
    环境抽象类
    """
    @abstractmethod
    def observation_dim(self) -> int:
        """状态空间维度"""

    @abstractmethod
    def action_dim(self) -> int:
        """动作空间维度"""

    @abstractmethod
    def is_discrete(self) -> bool:
        """是否是离散动作"""

    @abstractmethod
    def action_radius(self):
        """
        动作空间半径，算法会根据动作空间半径输出 [-radius, radius] 范围的动作值。
        更复杂的动作映射可以自定义 standard_action 方法。
        """

    @abstractmethod
    def action_high(self):
        """动作最大值"""

    @abstractmethod
    def action_low(self):
        """动作最小值"""

    @abstractmethod
    def standard_action(self, action):
        """标准化动作"""

    @abstractmethod
    def reset(self):
        """
        重置、启动环境
        返回：状态
        """

    @abstractmethod
    def step(self, action):
        """
        执行动作，更新环境状态
        返回：状态、奖励、是否结束、额外信息
        """
