#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/10.
"""
import os.path
from datetime import datetime

import numpy as np
import tensorflow as tf2

from tools import make_sure_dir

tf = tf2.compat.v1
tf.disable_v2_behavior()


LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # 经验池容量
BATCH_SIZE = 32  # 一批训练的大小
MODEL_SAVE_PATH = "./data/model"  # 模型保存路径
LOG_SAVE_PATH = "./data/log"  # 日志保存路径


class DDPG(object):
    """
    DDPG
    Actor Critic based algorithm.
    """
    def __init__(self, a_dim, s_dim, a_bound):
        """
        :param a_dim: 动作空间维度
        :param s_dim: 状态空间维度
        :param a_bound: 动作空间上下限，ddpg默认输出 [-1, 1] 范围，通过此参数可放大范围。若有更复杂的需求，请自行对action进行变换。
        """
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)

        self.saver = tf.train.Saver()
        self.create_time = int(datetime.now().timestamp())

        tensorboard_log_path = f'{LOG_SAVE_PATH}/tensorboard-{self.create_time}'
        self.summary_writer = tf.summary.FileWriter(tensorboard_log_path, graph=tf.get_default_graph(), flush_secs=10)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        self._bs = bt[:, :self.s_dim]
        self._ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        self._br = bt[:, -self.s_dim - 1: -self.s_dim]
        self._bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: self._bs})
        self.sess.run(self.ctrain, {self.S: self._bs, self.a: self._ba, self.R: self._br, self.S_: self._bs_})

    def cal_loss(self):
        """必须在learn之后调用"""
        loss = self.sess.run(self.a_loss, {self.S: self._bs, self.a: self._ba, self.R: self._br, self.S_: self._bs_})
        return loss

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save_model(self, m_name=None, f_name='network', **kwargs):
        if m_name is None:
            m_name = f'model-{self.create_time}'
        m_path = os.path.abspath(f'{MODEL_SAVE_PATH}/{m_name}')
        make_sure_dir(m_path)
        save_path = self.saver.save(self.sess, f'{m_path}/{f_name}', **kwargs)
        print("Model saved in file: %s" % save_path)
        return True

    def load_model(self, m_name=None):
        if not os.path.exists(MODEL_SAVE_PATH):
            return False
        if m_name is None:
            models = [(m, m.split('-')[-1]) for m in os.listdir(MODEL_SAVE_PATH)]
            models = [(m[0], int(m[1])) for m in models if m[1].isdigit()]
            if len(models) == 0:
                return False
            models.sort(key=lambda x: x[1])
            m_name = models[-1][0]

        latest_cp = tf.train.latest_checkpoint(f'{MODEL_SAVE_PATH}/{m_name}')
        if latest_cp is None:
            return False
        m_path = os.path.abspath(latest_cp)
        self.saver.restore(self.sess, m_path)
        print("Model restored from file: %s" % MODEL_SAVE_PATH)
        return True

    def print_tensorboard(self, value_dict, step):
        """
        打印到tensorboard
        :param value_dict: 标签名 -> 标量数据 的字典
        :param step: 步数
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=n, simple_value=v) for n, v in value_dict.items()])
        self.summary_writer.add_summary(summary, step)

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
