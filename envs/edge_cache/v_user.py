#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/13.
"""
import random

from envs.edge_cache.v_server import VirtualServer
from envs.edge_cache.v_station import VirtualStation, RandomChangeValue

random.seed(71)


class VirtualUser:
    # TODO: 用户终端播放速率？应该是定值。。。
    PLAY_SPEED = 0.1
    # TODO: 用户忍受的时延
    TOLERABLE_DELAY = 3

    def __init__(self, r0, r0_r=0.1):
        self.looked_videos = []
        self.r0 = RandomChangeValue(r0, r0_r, r0_r)
        self.looked_list_size = 10

    def play(self, virtual_station: VirtualStation):
        # 以50%概率看看过的视频
        look_new = 1 if random.random() > 0.5 else 0
        if len(self.looked_videos) == 0:
            look_new = 1
        if look_new:
            video = random.choice(VirtualServer.video_list)
            self.looked_videos.append(video)
            while len(self.looked_videos) > self.looked_list_size:
                self.looked_videos.pop(0)
        else:
            video = self.looked_videos[random.randint(0, len(self.looked_videos) - 1)]

        # 更新环境状态
        video.view += 1
        if look_new:
            if random.random() < 0.25:
                video.like += 1
            if random.random() < 0.01:
                video.share += 1
            if random.random() < 0.2:
                video.comment += 1
        else:
            if random.random() < 0.5:
                video.like += 1
            if random.random() < 0.08:
                video.share += 1
            if random.random() < 0.4:
                video.comment += 1

        cache_r_t, v_r_t = virtual_station.get_video(video.id, self.r0.get_now_value())
        return cache_r_t, v_r_t
