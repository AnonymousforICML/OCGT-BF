#!/usr/bin/env python
# coding=utf-8
from nda.optimizers import Optimizer
from nda.optimizers import compressor as cop


class DGD_tracking(Optimizer):
    '''The distributed gradient descent algorithm with gradient tracking, described in 'Harnessing Smoothness to Accelerate Distributed Optimization', Guannan Qu, Na Li'''

    def __init__(self, p, dim, eta=0.1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.grad_last = None
        self.dim = dim

    def init(self):
        super().init()
        self.s = self.grad(self.x)  # 初始化跟踪梯度变量s为当前梯度
        self.grad_last = self.s.copy()  # 初始化上一次迭代的梯度为当前梯度的副本

    def update(self):
        self.comm_rounds += 2

        self.x = self.x.dot(self.W) - self.eta * self.s
        grad_current = self.grad(self.x)

        self.trans_bits += (32 * self.dim)

        self.s = self.s.dot(self.W) + grad_current - self.grad_last
        self.trans_bits += (32 * self.dim)
        self.grad_last = grad_current  # 更新上一次迭代的梯度为当前梯度

