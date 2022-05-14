#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/11.
"""
import random

import numpy as np

from envs import AbsEnv
from envs.edge_cache.v_server import VirtualServer, Video
from envs.edge_cache.v_station import VirtualStation
from envs.edge_cache.v_user import VirtualUser
from tools import show_run_time


class SmallVideoEdgeCacheEnv(AbsEnv):
    STEP_USER_REQUERY = 100
    STEP_USER_REQUERY_FLUCTUATION = 0.1

    def __init__(self):
        self.v_station = VirtualStation(2500, 150, 10)
        self.user_num = 100
        b_avg = self.v_station.station_bandwidth / self.user_num
        self.users = []
        for i in range(self.user_num):
            self.users.append(VirtualUser(random.gauss(b_avg * 0.9, b_avg * 0.2)))

    def observation_dim(self) -> int:
        """状态空间维度"""
        return len(self._get_obs())

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
        return 1.0

    def action_high(self):
        """动作最大值"""

    def action_low(self):
        """动作最小值"""

    def standard_action(self, action):
        """标准化动作"""
        q, xn = action
        # q: 0-100离散值，xn: 0-5000离散值
        q = int((q + 1.0) * 100 / 2)
        if q < 0:
            q = 0
        elif q > 100:
            q = 100
        xn = int((xn + 1.0) * 5000 / 2)
        if xn < 0:
            xn = 0
        elif xn > 5000:
            xn = 5000
        return np.array([q, xn])

    def reset(self):
        """
        重置、启动环境
        返回：状态
        """
        self.v_station = VirtualStation(2500, 150, 10)
        self.user_num = 100
        b_avg = self.v_station.station_bandwidth / self.user_num
        self.users = []
        for i in range(self.user_num):
            self.users.append(VirtualUser(random.gauss(b_avg * 0.9, b_avg * 0.2)))

        VirtualServer.reset_all()

        return self._get_obs()

    @show_run_time()
    def step(self, action):
        """
        执行动作，更新环境状态
        返回：状态、奖励、是否结束、额外信息

        每一次step，user向基站请求 STEP_USER_REQUERY 次，评估这段时间的状态与奖励。
        """
        q, xn = action
        self.v_station.cache_q = q
        self.v_station.cache_xn = xn
        i = 0
        all_i = len(self.users) * self.STEP_USER_REQUERY
        for u in self.users:
            req_num = int(random.gauss(self.STEP_USER_REQUERY, self.STEP_USER_REQUERY_FLUCTUATION))
            for _ in range(req_num):
                u.play(self.v_station)

                i += 1
                print(f'\r{(i / all_i):.2}', end='')
        print()

        return self._get_obs(), 0, False, {}

    def _get_obs(self):
        """获取当前状态"""
        r0_avg = self.v_station.r0_all / self.v_station.round_step
        r1_avg = VirtualServer.r1_all / VirtualServer.round_step

        obs = [
            # 下行速率，从基站到用户
            self.users[0].r0.value if r0_avg == 0 else r0_avg,
            self.users[0].r0.value if self.v_station.r0_max == -float('inf') else self.v_station.r0_max,
            self.users[0].r0.value if self.v_station.r0_min == float('inf') else self.v_station.r0_min,
            # 回程速率，从服务器到基站
            self.v_station.r1.value if r1_avg == 0 else r1_avg,
            self.v_station.r1.value if VirtualServer.r1_max == -float('inf') else VirtualServer.r1_max,
            self.v_station.r1.value if VirtualServer.r1_min == float('inf') else VirtualServer.r1_min,
            # 用户终端播放速率
            VirtualUser.PLAY_SPEED,
            # 视频分块长度
            Video.VIDEO_PART_SIZE,
            # 用户忍受的时延
            VirtualUser.TOLERABLE_DELAY,
            # 基站缓存大小
            self.v_station.cache_size,
            # 该基站的带宽，从基站到用户的总带宽
            self.v_station.station_bandwidth,
            # 基站对应用户数
            self.user_num,
            # TODO: 剩余允许卡顿次数?
        ]
        return np.array(obs)
