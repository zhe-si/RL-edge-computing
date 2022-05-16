#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/13.
"""
import heapq
import random

from envs.edge_cache.v_server import VirtualServer, GlobalClock

random.seed(71)


class RandomChangeValue:
    """范围内随机浮动值"""
    def __init__(self, value, radius_low, radius_high, change_rate=1):
        """
        :param value: 初始值
        :param radius_low: 向下最大浮动百分比
        :param radius_high: 向上最大浮动百分比
        :param change_rate: 浮动变化速率，0-1之间
        """
        self.value = value
        self.radius_low = -abs(radius_low)
        self.radius_high = abs(radius_high)
        self.now_percentage = 1.0
        self.change_rate = change_rate

    def get_now_value(self):
        u_d = random.choice([-1, 1])
        abs_radius = self.radius_high if u_d == 1 else abs(self.radius_low)
        self.now_percentage *= 1 + u_d * abs(random.gauss(0, abs_radius)) * self.change_rate
        if 1 + self.radius_low > self.now_percentage or self.now_percentage > 1 + self.radius_high:
            self.now_percentage = 1.0
        return self.value * self.now_percentage


class FIFOCache:
    def __init__(self, cache_size, func_size, func_sort_key):
        self.cache_size = cache_size
        self.cache = {}
        self.cache_sort_heap = []

        self.func_size = func_size
        self.func_sort_key = func_sort_key
        self.now_size = 0

    def push(self, key, value):
        if key in self.cache:
            self.now_size -= self.func_size(self.cache[key])
            self.cache.pop(key)
        value_size = self.func_size(value)
        if value_size > self.cache_size:
            print('INFO: value is tool big, no enough space to push')
            return
        while self.now_size + value_size > self.cache_size:
            if len(self.cache_sort_heap) == 0:
                print('WARNING: value is tool big, no enough space to push')
                return
            sort_key, oldest_v_key = heapq.heappop(self.cache_sort_heap)
            v = self.cache.pop(oldest_v_key)
            self.now_size -= self.func_size(v)
        self.cache[key] = value
        heapq.heappush(self.cache_sort_heap, (self.func_sort_key(value), key))
        self.now_size += value_size

    def _remove_old(self):
        """删除比较值小的一个"""
        v_l = list(self.cache.items())
        v_l.sort(key=lambda v: self.func_sort_key(v[1]))
        self.cache.pop(v_l[0][0])
        self.now_size -= self.func_size(v_l[0][1])

    def get(self, key):
        return self.cache.get(key, None)

    def migrate_cache(self, new_cache):
        v_l = list(self.cache.items())
        v_l.sort(key=lambda _v: self.func_sort_key(_v[1]), reverse=True)
        for v in v_l:
            v_size = self.func_size(v)
            if v_size + new_cache.now_size > new_cache.cache_size:
                break
            new_cache.push(v[0], v[1])


class VirtualStation:
    def __init__(self, cache_size, bandwidth, r1, r1_r=0.1):
        """
        初始化
        :param cache_size: 缓存总大小
        :param bandwidth: 带宽
        :param r1: 回程速率基准值
        :param r1_r: 回程速率浮动值
        """
        self.r1 = RandomChangeValue(r1, r1_r, r1_r)

        # 基站缓存总大小
        self.cache_size = cache_size
        # 基站总带宽
        self.station_bandwidth = bandwidth

        # FIFO被动缓存，id -> (Video, last_access_time)
        self.video_passive_cache = FIFOCache(self.cache_size,
                                             lambda v: self._cal_video_cache_size(v[0]),
                                             lambda v: v[1])

        # 部分缓存块数
        self.cache_q = 5
        # 主动缓存文件数
        self.cache_xn = 0

        # 主动缓存
        # 受欢迎视频排名
        self.popularity_videos = []
        # 当前主动缓存大小
        self.active_cache_size = 0
        # 主动缓存的视频
        self.cached_popularity_videos = {}
        # 所有流行视频
        self.cached_popularity_videos_all = {}
        self.update_popularity_videos()

        # 每轮次统计信息
        self.round_step = 1
        # 请求次数
        self.req_num = 1
        # 命中次数
        self.hit_num = 0
        # 每轮下行速率统计
        self.r0_all = 0
        self.r0_min = float('inf')
        self.r0_max = -float('inf')

    def _update_r0(self, r0):
        self.r0_all += r0
        self.r0_min = min(self.r0_min, r0)
        self.r0_max = max(self.r0_max, r0)

    @staticmethod
    def _popularity(video):
        """计算视频受欢迎度"""
        a1, a2, a3, a4 = 0.5, 1 / 3, 1 / 12, 1 / 12
        return a1 * video.view + a2 * video.like + a3 * video.share + a4 * video.comment

    def update_passive_cache(self, is_migrate=False):
        """
        更新被动缓存，并清空被动缓存
        :param is_migrate: 是否从旧的缓存迁移数据
        """
        video_passive_cache = FIFOCache(self.cache_size - self.active_cache_size,
                                        lambda _v: self._cal_video_cache_size(_v[0]),
                                        lambda _v: _v[1])
        if is_migrate:
            self.video_passive_cache.migrate_cache(video_passive_cache)
        self.video_passive_cache = video_passive_cache

    def update_popularity_videos(self):
        """更新受欢迎视频排名"""
        self.popularity_videos = sorted(VirtualServer.video_list, key=lambda _v: self._popularity(_v), reverse=True)
        self.active_cache_size = 0
        self.cached_popularity_videos = {}
        self.cached_popularity_videos_all = {}
        for i in range(self.cache_xn):
            v = self.popularity_videos[i]
            self.active_cache_size += self._cal_video_cache_size(v)
            self.cached_popularity_videos[v.id] = (v, i + 1)
        self.cached_popularity_videos_all = self.cached_popularity_videos.copy()
        for i in range(self.cache_xn, len(self.popularity_videos)):
            v = self.popularity_videos[i]
            self.cached_popularity_videos_all[v.id] = (v, i + 1)

    def reset_statistics(self):
        """重置统计信息"""
        self.round_step = 1
        self.req_num = 1
        self.hit_num = 0
        self.r0_all = 0
        self.r0_min = float('inf')
        self.r0_max = -float('inf')

    @staticmethod
    def _get_video_from_list(v_list, v_id):
        for v in v_list:
            if v.id == v_id:
                return v
        return None

    def get_video(self, v_id, r0):
        """
        模拟用户向基站请求视频
        :param v_id: 视频id
        :param r0: 基站到用户的回程速率
        :return: (缓存返回时间，剩余数据请求并返回时间)
        """
        self.req_num += 1
        self.round_step += 1
        self._update_r0(r0)

        if v_id in self.cached_popularity_videos:
            v, _ = self.cached_popularity_videos[v_id]
            self.hit_num += 1
            return self._cal_cost(r0, v_id, v)
        elif v_id in self.video_passive_cache.cache:
            self.hit_num += 1
            video, last_access_time = self.video_passive_cache.get(v_id)
            return self._cal_cost(r0, v_id, video)
        else:
            # 主动和被动缓存都没命中，从服务器获取，并缓存到被动缓存
            video = VirtualServer.video_list[v_id]
            self.video_passive_cache.push(v_id, (video, GlobalClock.get_now_time()))
            return 0.0, VirtualServer.get_video(v_id, self.r1.get_now_value(), 0) + video.size / r0

    def _cal_cost(self, r0, v_id, video):
        """计算传输花费的时间"""
        save_part_num = self._cal_save_part_num(video)
        cache_all = save_part_num >= video.part_num
        cache_s = self._cal_video_cache_size(video)
        cache_t = cache_s / r0
        if cache_all:
            return cache_t, 0.0
        else:
            return cache_t, VirtualServer.get_video(v_id, self.r1.get_now_value(), save_part_num) + (
                    video.size - cache_s) / r0

    def _cal_video_cache_size(self, video):
        """基于缓存视频段数计算缓存视频大小"""
        save_part_num = self._cal_save_part_num(video)
        cache_all = save_part_num >= video.part_num
        cache_s = save_part_num * video.VIDEO_PART_SIZE if not cache_all else video.size
        return cache_s

    def _cal_save_part_num(self, video):
        """计算缓存视频的段数"""
        return self.cache_q if video.part_num > self.cache_q else video.part_num
