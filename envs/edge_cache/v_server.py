#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/13.
"""
import copy
import random
import math


random.seed(71)


class GlobalClock:
    """模拟一个全局时钟"""
    now_time = 0

    @classmethod
    def get_now_time(cls):
        cls.now_time += 1
        return cls.now_time


class Video:
    # 从1开始编号
    now_id = 0
    # 假设60s视频对应内存大小为1MB
    SIZE_RANGE = (5 / 60, 3)
    VIDEO_PART_SIZE = 0.2

    @classmethod
    def create_video(cls):
        cls.now_id += 1
        return Video(cls.now_id)

    def __init__(self, v_id):
        self.id = v_id
        self.size = random.uniform(self.SIZE_RANGE[0], self.SIZE_RANGE[1])
        self.part_num = math.ceil(self.size / self.VIDEO_PART_SIZE)
        self.last_part_size = self.size - (self.part_num - 1) * self.VIDEO_PART_SIZE

        # 播放量
        self.view = int(random.gauss(10 * 10000, 50000))
        # 点赞量
        self.like = int(random.gauss(1.5 * 10000, 10000))
        # 转发量
        self.share = int((self.view + 5 * self.like) / 240 * random.gauss(1.0, 0.4))
        # 评论量
        self.comment = self.share + self.like * random.gauss(1.0, 0.2) + self.view * random.gauss(0.1, 0.1)
        if random.random() < 0.2:
            self.view *= random.gauss(0.25, 0.1)
            self.like *= random.gauss(0.25, 0.1)
            self.share *= random.gauss(0.25, 0.1)
            self.comment *= random.gauss(0.25, 0.1)

        # 视频分类
        # self.type = -1


class VirtualServer:
    video_list = [Video.create_video() for _ in range(50000)]
    video_list_copy = copy.deepcopy(video_list)

    # 每轮次统计信息
    round_step = 0
    # 每轮回程速率统计
    r1_all = 0
    r1_min = float('inf')
    r1_max = -float('inf')

    @classmethod
    def reset_all(cls):
        cls.reset_statistics()
        cls.video_list = copy.deepcopy(cls.video_list_copy)

    @classmethod
    def reset_statistics(cls):
        """重置统计信息"""
        cls.round_step = 0
        cls.r0_all = 0
        cls.r0_min = float('inf')
        cls.r0_max = -float('inf')

    @classmethod
    def _update_r1(cls, r1):
        cls.r1_all += r1
        cls.r1_min = min(cls.r1_min, r1)
        cls.r1_max = max(cls.r1_max, r1)

    @classmethod
    def add_video(cls, v_num):
        cls.video_list.extend([Video.create_video() for _ in range(v_num)])

    @classmethod
    def get_video(cls, v_id, r1, start_part):
        """
        模拟基站请求视频
        :param v_id: 视频id
        :param r1: 回程速率
        :param start_part: 起始视频段编号，视频段编号从0开始
        :return: 传输视频所用时间
        """
        cls.round_step += 1
        cls._update_r1(r1)

        v = cls.video_list[v_id - 1]
        if start_part >= v.part_num:
            return 0
        return ((v.part_num - start_part - 1) * Video.VIDEO_PART_SIZE + v.last_part_size) / r1
