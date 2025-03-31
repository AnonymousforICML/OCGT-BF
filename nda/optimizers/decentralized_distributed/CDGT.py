#!/usr/bin/env python
# coding=utf-8
from nda.optimizers import Optimizer
from nda.optimizers import compressor as cop
import numpy as np

class CDGT(Optimizer):
    """
    The Compressed Distributed Gradient Tracking (CDGT) algorithm.

    更新公式（单次迭代）：

    1. 对 x 的更新

    $$
    q^x = Q\bigl(x_t - \hat{h}^x_t\bigr)
    $$
    $$
    \tilde{x}_t = \tilde{h}^x_t + q^x W
    $$
    $$
    \hat{x}_t = \hat{h}^x_t + q^x W
    $$
    $$
    \hat{h}^x_{t+1} = \alpha\,\hat{x}_t + (1-\alpha)\,\hat{h}^x_t
    $$
    $$
    \tilde{h}^x_{t+1} = \alpha\,\tilde{x}_t + (1-\alpha)\,\tilde{h}^x_t
    $$
    $$
    x_{t+1}=x_t+0.5\bigl[\hat{h}^x_t-\tilde{h}^x_t\bigr]-\eta\,y_t
    $$

    2. 对 y 的更新

    $$
    q^y = Q\bigl(y_t - \hat{h}^y_t\bigr)
    $$
    $$
    \tilde{y}_t = \tilde{h}^y_t + q^y W
    $$
    $$
    \hat{y}_t = \hat{h}^y_t + q^y W
    $$
    $$
    \hat{h}^y_{t+1} = \alpha\,\hat{y}_t + (1-\alpha)\,\hat{h}^y_t
    $$
    $$
    \tilde{h}^y_{t+1} = \alpha\,\tilde{y}_t + (1-\alpha)\,\tilde{h}^y_t
    $$
    $$
    y_{t+1}=y_t+0.5\bigl[\hat{h}^y_t-\tilde{h}^y_t\bigr] + g_{t+1} - g_t
    $$

    这里:
    - $Q(\cdot)$ 表示量化（压缩）操作；
    - $\hat{h}^x_t,\tilde{h}^x_t,\hat{h}^y_t,\tilde{h}^y_t$ 为算法的内部辅助变量；
    - $W$ 为网络拓扑对应的混合矩阵 (Metropolis-Hastings 或者任意满足共轭性的权重矩阵)；
    - $g_{t+1}-g_t$ 用于模拟梯度的追踪更新（在实际实现里需根据需求获取或计算梯度）；
    - $\alpha \in (0,1]$ 是更新中用于平滑/记忆的参数；
    - $\eta>0$ 是步长 (learning rate)。
    """

    def __init__(self, 
                 p,              # Problem / parent class
                 dim,            # 变量 x, y 的维度
                 alpha=0.5,      # 平滑/记忆参数，默认为0.5
                 eta=0.1,        # 学习率
                 s=10,           # 初始量化等级
                 n_agent=0,      # 节点数
                 W=None,         # 混合矩阵
                 x_0=None,       # 初始值
                 n_iters=100,    # 迭代次数
                 name='CDGT',    # 算法名称
                 **kwargs):
        super().__init__(p, **kwargs)
        self.alpha = alpha
        self.eta = eta
        self.dim = dim
        self.s = s
        self.n_agent = n_agent
        self.W = W
        self.x_0 = x_0
        self.n_iters = n_iters
        self.name = name

        # CDGT 特有的一些内部状态变量
        self.hat_hx = None
        self.tilde_hx = None
        self.hat_hy = None
        self.tilde_hy = None

        # 记录当前及上一时刻梯度
        self.grad_last = None
        self.grad_current = None

    def init(self):
        """
        初始化算法在迭代过程所需的状态。
        """
        super().init()  # 父类初始化，通常会初始化 self.x

        # 初始化 y 为初始梯度（而不是零矩阵）
        self.y = self.grad(self.x).copy()
        
        # 初始化辅助变量为当前值（而不是零矩阵）
        self.hat_hx = self.x.copy()
        self.tilde_hx = self.x.copy()
        self.hat_hy = self.y.copy()
        self.tilde_hy = self.y.copy()

        # 初始化梯度
        self.grad_last = self.y.copy()  # 使用 y 作为初始梯度
        self.grad_current = self.grad_last.copy()

    def update(self):
        """
        根据 CDGT 的更新公式进行一次迭代。
        """
        self.comm_rounds += 2
        self.s += 1

        ##################################################
        # 1) 针对 x 的更新
        ##################################################
        Q_x = np.zeros((self.dim, self.n_agent))
        bits_x = 0
        
        # 计算压缩误差
        diff_x = self.x - self.hat_hx
        for i in range(self.n_agent):
            result = cop.Deterministic_quantization(
                x=diff_x[:, i],
                s=self.s
            )
            Q_x[:, i], bits_i = result[0], result[1]
            bits_x += bits_i
        
        self.trans_bits += bits_x

        # 更新 x 相关变量
        tilde_x = self.tilde_hx + Q_x
        hat_x = self.hat_hx + Q_x.dot(self.W)
        
        # 修改 x 的更新方式，参考 QDGT 的更新方式
        x_new = self.x + 0.5 * (hat_x - tilde_x) - self.eta * self.y
        
        # 辅助变量更新
        hat_hx_new = self.alpha * hat_x + (1 - self.alpha) * self.hat_hx
        tilde_hx_new = self.alpha * tilde_x + (1 - self.alpha) * self.tilde_hx

        ##################################################
        # 2) 针对 y 的更新
        ##################################################
        Q_y = np.zeros((self.dim, self.n_agent))
        bits_y = 0
        
        # 计算压缩误差
        diff_y = self.y - self.hat_hy
        for i in range(self.n_agent):
            result = cop.Deterministic_quantization(
                x=diff_y[:, i],
                s=self.s
            )
            Q_y[:, i], bits_i = result[0], result[1]
            bits_y += bits_i
        
        self.trans_bits += bits_y

        # 更新 y 相关变量
        tilde_y = self.tilde_hy + Q_y
        hat_y = self.hat_hy + Q_y.dot(self.W)
        
        # 计算新梯度
        g_current = self.grad(x_new)
        
        # 修改 y 的更新方式，参考 QDGT 的更新方式
        y_new = self.y + 0.5 * (hat_y - tilde_y) + g_current - self.grad_last
        
        # 辅助变量更新
        hat_hy_new = self.alpha * hat_y + (1 - self.alpha) * self.hat_hy
        tilde_hy_new = self.alpha * tilde_y + (1 - self.alpha) * self.tilde_hy

        ##################################################
        # 3) 更新所有状态变量
        ##################################################
        self.x = x_new
        self.y = y_new
        self.hat_hx = hat_hx_new
        self.tilde_hx = tilde_hx_new
        self.hat_hy = hat_hy_new
        self.tilde_hy = tilde_hy_new
        self.grad_last = g_current