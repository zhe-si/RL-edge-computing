#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/14.
"""
import numpy as np


# 探索策略
def normal_explore(a, e):
    """
    正态分布探索
    以输出的a为中心，以e为标准差，生成一个符合正态分布的随机数(探索)
    """
    return np.random.normal(a, e)


def uniform_explore(a, e):
    """
    均匀分布探索
    以输出的a为中心，以e为半径，生成一个符合均匀分布的随机数(探索)
    """
    return np.random.uniform(a - e / 2, a + e / 2)


# 更新探索程度策略
def exponential_decay(decay_rate=0.9995):
    """
    指数衰减更新探索程度
    """
    return lambda _e: _e * decay_rate
