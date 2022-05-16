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

random.seed(71)


class SmallVideoEdgeCacheEnv(AbsEnv):
    STEP_USER_REQUERY = 100
    STEP_USER_REQUERY_FLUCTUATION = 0.1 * STEP_USER_REQUERY

    def __init__(self):
        self.v_station = VirtualStation(2000, 20, 2.8)
        self.user_num = 100
        # b_avg = self.v_station.station_bandwidth / self.user_num
        self.users = []
        for i in range(self.user_num):
            # self.users.append(VirtualUser(random.gauss(b_avg * 0.9, b_avg * 0.2)))
            self.users.append(VirtualUser(1.5))

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
        # q: 1-101离散值，xn: 0-5000离散值
        q = int((q + 1.0) * 100 / 2 + 1)
        if q < 1:
            q = 1
        elif q > 101:
            q = 101
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
        self.v_station = VirtualStation(2000, 20, 2.8)
        self.user_num = 100
        self.users = []
        for i in range(self.user_num):
            self.users.append(VirtualUser(1.5))

        VirtualServer.reset_all()

        return self._get_obs()

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
        max_v_r_t = -1
        max_cache_r_t = -1
        for u in self.users:
            req_num = int(random.gauss(self.STEP_USER_REQUERY, self.STEP_USER_REQUERY_FLUCTUATION))
            for _ in range(req_num):
                cache_r_t, v_r_t = u.play(self.v_station)
                if max_v_r_t < v_r_t:
                    max_v_r_t = v_r_t
                if max_cache_r_t < cache_r_t:
                    max_cache_r_t = cache_r_t

                i += 1
                # print(f'\r{(i / all_i):.2}', end='')
        # print()

        done, punish = self._check_is_done(max_v_r_t, max_cache_r_t)
        if punish is not None:
            utility = punish
        else:
            utility = self.cal_reward()

        print(f'{self.v_station.cache_q:3}, {self.v_station.cache_xn:4}: \t{(self.v_station.hit_num / self.v_station.req_num):.6}, \t{utility:.6}')

        return self._get_obs(), utility, done, {}

    def _check_is_done(self, max_v_r_t, max_cache_r_t):
        """
        返回 done、punish(若无惩罚，返回None)
        """
        # 主动缓存不可大于基站缓存空间
        if self.v_station.active_cache_size > self.v_station.cache_size:
            return True, -3
        # 某视频缓存大小 <= 单个视频缓存最大段数 * 每段长度
        # 单个视频最大缓存传输到用户的时间不可大于用户忍耐时间
        if max_cache_r_t > VirtualUser.TOLERABLE_DELAY:
            return True, -3
        # 主动与被动缓存的最大视频的下载时间不可大于缓存视频的最大播放时间
        if max_v_r_t > self.v_station.cache_q * Video.VIDEO_PART_SIZE / VirtualUser.PLAY_SPEED:
            return True, -3
        if self.v_station.cache_q > Video.SIZE_RANGE[1] / VirtualUser.PLAY_SPEED:
            return True, -3
        return False, None

    def cal_reward(self):
        PA = 0.552
        fenmu = 0
        for v in VirtualServer.video_list:
            fenmu = fenmu + self.v_station.cached_popularity_videos_all[v.id][1] ** (-PA)
        utility = 0
        for v in VirtualServer.video_list:
            if v.id in self.v_station.cached_popularity_videos or self.v_station.video_passive_cache.get(v.id) is not None:
                utility = utility + self.v_station.cached_popularity_videos_all[v.id][1] ** (-PA) / fenmu
        return utility

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
