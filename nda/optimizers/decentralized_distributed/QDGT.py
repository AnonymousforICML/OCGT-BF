#!/usr/bin/env python
# coding=utf-8
from nda.optimizers import Optimizer
from nda.optimizers import compressor as cop
import numpy as np

class QDGT(Optimizer):
    '''The distributed gradient descent algorithm with gradient tracking, described in 'Harnessing Smoothness to Accelerate Distributed Optimization', Guannan Qu, Na Li'''

    def __init__(self, p, dim, eta=0.1,s=10, n_agent=0, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.grad_last = None
        self.dim = dim
        self.s = s  # 量化等级
        self.n_agent = n_agent

    def init(self):
        super().init()
        self.grad_y = self.grad(self.x)  # 初始化跟踪梯度变量s为当前梯度
        self.grad_last = self.grad_y.copy()  # 初始化上一次迭代的梯度为当前梯度的副本

    def update(self):
        self.comm_rounds += 2
        Q_x = np.zeros((self.dim, self.n_agent))
        self.s = self.s+1
        bits = 0

        for i in range(self.n_agent):
            Q_x[:, i], bits = cop.Deterministic_quantization(x=self.x[:, i], s=self.s)

        self.x = self.x+0.5*(Q_x.dot(self.W)-Q_x) - self.eta * self.grad_y

        grad_current = self.grad(self.x)

        self.trans_bits += bits

        Q_grad_y = np.zeros((self.dim, self.n_agent))
        for i in range(self.n_agent):
            Q_grad_y[:, i], bits = cop.Deterministic_quantization(x=self.grad_y[:, i], s=self.s)  # 只需获得一个agent传输bits即可
        self.trans_bits += bits

        self.grad_y = self.grad_y + 0.5 * (Q_grad_y.dot(self.W) - Q_grad_y) + grad_current - self.grad_last

        self.grad_last = grad_current  # 更新上一次迭代的梯度为当前梯度

