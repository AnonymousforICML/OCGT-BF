#!/usr/bin/env python
# coding=utf-8
from nda.optimizers import Optimizer
from nda.optimizers import compressor as cop
import numpy as np

class CDGT_bandit_decay_randk(Optimizer):


    def __init__(self, 
                 p,              # Problem / parent class
                 dim,            # 变量 x, y 的维度
                 alpha=0.5,      # 平滑/记忆参数，默认为0.5
                 eta=0.01,        # 学习率
                 s=10,           # 初始随机选择的元素个数
                 n_agent=0,      # 节点数
                 W=None,         # 混合矩阵
                 x_0=None,       # 初始值
                 n_iters=100,    # 迭代次数
                 name='CDGT_bandit',    # 算法名称
                 mu=0.9,         # 初始mu值调大到1.0
                 mu_min=1e-3,    # 最小值也相应调大
                 decay_rate=0.98, # 衰减率调大，使衰减更平缓
                 decay_steps=20,  # 增加衰减步长，使衰减更平滑
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
        self.mu = mu
        self.mu_min = mu_min
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_iter = 0

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
        super().init()

        # 使用零阶梯度估计初始化y
        self.y = self.zeroth_order_gradient(self.x).copy()
        
        # 初始化辅助变量
        self.hat_hx = self.x.copy()
        self.tilde_hx = self.x.copy()
        self.hat_hy = self.y.copy()
        self.tilde_hy = self.y.copy()

        # 初始化梯度
        self.grad_last = self.y.copy()
        self.grad_current = self.grad_last.copy()

    def update(self):
        """
        根据 CDGT 的更新公式进行一次迭代。
        """
        self.current_iter += 1
        
        # 更新mu值
        if self.current_iter % self.decay_steps == 0:
            self.mu = max(self.mu * self.decay_rate, self.mu_min)
            
        self.comm_rounds += 2

        ##################################################
        # 1) 针对 x 的更新
        ##################################################
        Q_x = np.zeros((self.dim, self.n_agent))
        
        # 计算压缩误差
        diff_x = self.x - self.hat_hx
        for i in range(self.n_agent):
            # 使用Random-k压缩
            Q_x[:, i] = cop.random(diff_x[:, i].copy(), self.s)
        
        # 计算非零元素数量 * 32（每个浮点数占用32位）
        self.trans_bits += self.s * 32

        # 更新 x 相关变量
        tilde_x = self.tilde_hx + Q_x
        hat_x = self.hat_hx + Q_x.dot(self.W)
        
        # x 的更新
        # 原来的更新可能太激进，导致步长效果不明显
        x_new = self.x + self.eta * ((self.hat_hx - self.tilde_hx) - self.y)
        
        # 辅助变量更新时确保alpha真正起作用
        hat_hx_new = self.alpha * hat_x + (1 - self.alpha) * self.hat_hx
        tilde_hx_new = self.alpha * tilde_x + (1 - self.alpha) * self.tilde_hx

        ##################################################
        # 2) 针对 y 的更新
        ##################################################
        Q_y = np.zeros((self.dim, self.n_agent))
        
        # 计算压缩误差
        diff_y = self.y - self.hat_hy
        for i in range(self.n_agent):
            # 使用Random-k压缩
            Q_y[:, i] = cop.random(diff_y[:, i].copy(), self.s)
        
        # 计算非零元素数量 * 32（每个浮点数占用32位）
        self.trans_bits += self.s * 32

        # 更新 y 相关变量
        tilde_y = self.tilde_hy + Q_y
        hat_y = self.hat_hy + Q_y.dot(self.W)
        
        # 计算新梯度
        g_current = self.zeroth_order_gradient(x_new)
        
        # y 的更新
        # 添加步长控制
        y_new = self.y + self.eta * ((self.hat_hy - self.tilde_hy) + (g_current - self.grad_last))
        
        # 辅助变量更新时确保alpha真正起作用
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

    def zeroth_order_gradient(self, x):
        """
        计算零阶梯度估计，添加数值稳定性处理
        """
        d = self.dim
        n = self.n_agent
        grad_est = np.zeros((d, n))
        
        for i in range(n):
            u = np.random.normal(0, 1, (d, 1))  
            # 避免除以零，添加小量
            norm = np.linalg.norm(u) + 1e-10
            u = u / norm           
            
            x_plus = x[:, i] + self.mu * u.flatten()
            x_minus = x[:, i] - self.mu * u.flatten()
            
            f_plus = self.p.f(x_plus, i=i)
            f_minus = self.p.f(x_minus, i=i)
            
            # 添加数值稳定性处理
            diff = f_plus - f_minus
            if np.isnan(diff) or np.isinf(diff):
                diff = 0.0
                
            grad_est[:, i] = (d * diff / (2 * self.mu)) * u.flatten()
            
            # 梯度裁剪，避免梯度爆炸
            grad_norm = np.linalg.norm(grad_est[:, i])
            if grad_norm > 1e3:
                grad_est[:, i] *= 1e3 / grad_norm
        
        return grad_est