#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/11.
"""
from envs import AbsEnv


class SmallVideoEdgeCacheEnv(AbsEnv):
    class Obs:
        def __init__(self):
            # 下行速率，从基站到用户
            self.r0 = -1
            # 回程速率，从服务器到基站
            self.r1 = -1
            # 用户终端播放速率
            self.rp = -1
            # 视频分块长度
            self.s = -1
            # 用户忍受的时延
            self.t0 = -1
            # 基站缓存大小
            self.station_cache_size = -1
            # 该基站的带宽，从基站到用户的总带宽
            self.station_b = -1
            # 基站对应用户数
            self.station_user_num = -1

            # # 播放量
            # self.view = -1
            # # 点赞量
            # self.like = -1
            # # 转发量
            # self.share = -1
            # # 评论量
            # self.comment = -1
            #
            # # 视频分类
            # self.type = -1

    def __init__(self):
        pass

    def observation_dim(self) -> int:
        """状态空间维度"""

    def action_dim(self) -> int:
        """动作空间维度"""
        return 2

    def is_discrete(self) -> bool:
        """是否是离散动作"""
        return False

    def action_radius(self):
        """
        动作空间半径(默认为1)，算法会根据动作空间半径输出 [-radius, radius] 范围的动作值。
        更复杂的动作映射可以自定义 standard_action 方法。
        """

    def action_high(self):
        """动作最大值"""

    def action_low(self):
        """动作最小值"""

    def standard_action(self, action):
        """标准化动作"""

    def reset(self):
        """
        重置、启动环境
        返回：状态
        """

    def step(self, action):
        """
        执行动作，更新环境状态
        返回：状态、奖励、是否结束、额外信息
        """

    def _get_obs(self):
        """获取当前状态"""
