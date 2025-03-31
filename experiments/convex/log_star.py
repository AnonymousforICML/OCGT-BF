#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from nda.problems import LogisticRegression
from nda.optimizers import *
from nda.optimizers.utils import generate_mixing_matrix

from nda.experiment_utils import run_exp

if __name__ == '__main__':
    n_agent = 10  # 代理的数量
    m = 641  # 每个代理的采样数
    dim = 124  # 变量的维数


    kappa = 10
    mu = 5e-10
    n_iters = 10

    p = LogisticRegression(n_agent=n_agent, m=m, dim=dim, dataset='a5a', noise_ratio=0.3, graph_type='star', kappa=kappa)
    print(p.n_edges)


    x_0 = np.random.rand(dim, n_agent)
    x_0_mean = x_0.mean(axis=1)
    W, alpha = generate_mixing_matrix(p)
    print('alpha = ' + str(alpha))


    eta = 2/(p.L + p.sigma)

    n_dgd_iters = n_iters * 100
    batch_size = int(m / 100)

    # 对比batch——size 通信轮数
    distributed1 = [
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.02, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=10,
                               name='Q-SGT,batch-size=1'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.02, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=10,
                               batch_size=5, name='Q-SGT,batch-size=5'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.02, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=10,
                               batch_size=20, name='Q-SGT,batch-size=20'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.02, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=10,
                               batch_size=m, name='Q-SGT,batch-size=500'),
    ]
    # 对比不同的量化等级 通信轮数
    distributed2 = [
        DGD_tracking(p, n_iters=n_dgd_iters, eta=0.015, x_0=x_0, W=W, dim=dim, name='DGT'),
        GT_SAGA(p, n_iters=n_dgd_iters, batch_size=20, eta=0.015, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim,
                name='GT-SAGA'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.015, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=1,
                               batch_size=20, name=r'Q-SGT,$B_0$=1'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.015, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=8,
                               batch_size=20, name=r'Q-SGT,$B_0$=8'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.015, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=32,
                               batch_size=20, name=r'Q-SGT,$B_0$=32'),
    ]

    # 收敛性
    distributed3 = [
        # DGD_tracking(p, n_iters=n_dgd_iters, eta=0.02, x_0=x_0, W=W, dim=dim, name='DGT'),
        QDGT(p, n_iters=n_dgd_iters, eta=0.02, s=8, x_0=x_0, W=W, n_agent=n_agent, dim=dim, name='Q-DGT,\u03B7=0.02'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.02, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=8,
                               batch_size=20,
                               name='Q-SGT,\u03B7=0.02'),
        QDGT(p, n_iters=n_dgd_iters, eta=0.015, s=8, x_0=x_0, W=W, n_agent=n_agent, dim=dim,
             name='Q-DGT,\u03B7=0.015'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.015, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=8,
                               batch_size=20,
                               name='Q-SGT,\u03B7=0.015'),
        QDGT(p, n_iters=n_dgd_iters, eta=0.01, s=8, x_0=x_0, W=W, n_agent=n_agent, dim=dim, name='Q-DGT,\u03B7=0.01'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.01, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=8,
                               batch_size=20,
                               name='Q-SGT,\u03B7=0.01')
    ]
    # 传输比特
    distributed4 = [
        DGD_tracking(p, n_iters=n_dgd_iters, eta=0.02, x_0=x_0, W=W, dim=dim, name='DGT'),
        Q_DSAGA_tracking_batch(p, n_iters=n_dgd_iters, eta=0.02, x_0=x_0, W=W, n_agent=n_agent, m=m, dim=dim, s=1,
                               batch_size=30,
                               name='Q-SGT')
    ]

    res = run_exp(distributed3, kappa=kappa, max_iter=n_iters, name='logistic_regression', n_cpu_processes=4, save=True)


    plt.show()
