#!/usr/bin/env python
# coding=utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from nda.problems import LogisticRegression
from nda.optimizers import *
from nda.optimizers.utils import generate_mixing_matrix

from nda.experiment_utils import run_exp

if __name__ == '__main__':
    n_agent = 30  # 代理的数量
    m = 30  # 每个代理的采样数
    dim = 58  # spam数据集的特征维度是57+1(偏置项)

    kappa = 10
    mu = 5e-10
    n_iters = 20

    p = LogisticRegression(
        n_agent=n_agent, 
        m=m, 
        dim=dim, 
        dataset='spam',
        noise_ratio=0.3, 
        graph_type='er',
        graph_params=0.4, 
        kappa=kappa
    )
    x_0 = np.random.rand(dim, n_agent)
    x_0_mean = x_0.mean(axis=1)
    W, alpha = generate_mixing_matrix(p)
    print('alpha = ' + str(alpha))


    eta = 2/(p.L + p.sigma)

    n_dgd_iters = n_iters * 10
    batch_size = int(m / 100)


    # Regret
    distributed1 = [
        QDGT(p, n_iters=n_dgd_iters, eta=0.02, s=10, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='Q-DGT,\u03B7=0.02'),
        CDGT(p, n_iters=n_dgd_iters, eta=0.01, alpha=0.9, s=10, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT,\u03B7=0.01,α=0.9'),
        CDGT(p, n_iters=n_dgd_iters, eta=0.01, alpha=0.7, s=10, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT,\u03B7=0.01,α=0.7'),
        CDGT(p, n_iters=n_dgd_iters, eta=0.01, alpha=0.5, s=10, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT,\u03B7=0.01,α=0.5'),
        CDGT_bandit(p, n_iters=n_dgd_iters, eta=0.001, alpha=0.6, s=10, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT_bandit,\u03B7=0.001,α=0.6'),
        CDGT_bandit(p, n_iters=n_dgd_iters, eta=0.001, alpha=0.5, s=10, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT_bandit,\u03B7=0.001,α=0.5'),     
        CDGT_bandit(p, n_iters=n_dgd_iters, eta=0.0015, alpha=0.3, s=10, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT_bandit,\u03B7=0.0015,α=0.3'),
        CDGT_bandit(p, n_iters=n_dgd_iters, eta=0.003, alpha=0.6, s=10, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT_bandit,\u03B7=0.003,α=0.6')
    ]
    # ACC
    distributed2 = [
        DGD_tracking(p, n_iters=n_dgd_iters, eta=0.005, x_0=x_0, W=W, dim=dim, name='DGT'),
        CDGT_bandit(p, n_iters=n_dgd_iters, eta=0.0025, alpha=0.6, s=5, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT_bandit,\u03B7=0.003,α=0.6'),
        CDGT(p, n_iters=n_dgd_iters, eta=0.0025, alpha=0.6, s=5, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT_sto,\u03B7=0.003,α=0.6'),
        CDGT_bandit_decay(p, n_iters=n_dgd_iters, eta=0.0025, alpha=0.6, s=5, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT_bandit_decay,\u03B7=0.003,α=0.6')
    ]
    # 传输比特
    distributed3 = [
        DGD_tracking(p, n_iters=n_dgd_iters, eta=0.005, x_0=x_0, W=W, dim=dim, name='DGT'),
        CDGT_bandit_decay(p, n_iters=n_dgd_iters, eta=0.0025, alpha=0.6, s=5, x_0=x_0, W=W, n_agent=n_agent, dim=dim, 
             name='CDGT_bandit_decay,\u03B7=0.003,α=0.6'),
        CDGT_bandit_decay_topk(p, n_iters=n_dgd_iters, eta=0.0020, alpha=0.6, s=12, x_0=x_0, W=W, n_agent=n_agent, dim=dim,
             name='CDGT_bandit_decay_topk,\u03B7=0.003,α=0.6'),
        CDGT_bandit_decay_randk(p, n_iters=n_dgd_iters, eta=0.0020, alpha=0.6, s=12, x_0=x_0, W=W, n_agent=n_agent, dim=dim,
             name='CDGT_bandit_decay_randk,\u03B7=0.003,α=0.6')
    ]

    res = run_exp(distributed3, kappa=kappa, max_iter=n_iters, name='logistic_regression', n_cpu_processes=4, save=True)


    plt.show()
