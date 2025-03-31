#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from nda.problems import LogisticRegression
from nda.optimizers import *
from nda.optimizers.utils import generate_mixing_matrix

from nda.experiment_utils import run_exp

if __name__ == '__main__':
    n_agent_1 = 10  # 代理的数量
    n_agent_2 = 10  # 代理的数量
    n_agent_3 = 100  # 代理的数量
    m_1 = 641  # 每个代理的采样数
    m_2 = 641  # 每个代理的采样数
    m_3 = 64  # 每个代理的采样数
    dim = 124  # 变量的维数


    kappa = 10
    mu = 5e-10
    n_iters = 10

    p_1 = LogisticRegression(n_agent=n_agent_1, m=m_1, dim=dim, dataset='a5a', noise_ratio=0.3, graph_type='cycle', kappa=kappa)
    W_1, alpha_1 = generate_mixing_matrix(p_1)
    p_2 = LogisticRegression(n_agent=n_agent_2, m=m_2, dim=dim, dataset='a5a', noise_ratio=0.3, graph_type='star',
                           kappa=kappa)
    W_2, alpha_2 = generate_mixing_matrix(p_2)
    p_3 = LogisticRegression(n_agent=n_agent_3, m=m_3, dim=dim, dataset='a5a', noise_ratio=0.3, graph_type='er',graph_params=0.1,
                           kappa=kappa)
    W_3, alpha_3 = generate_mixing_matrix(p_3)


    x_0_1 = np.random.rand(dim, n_agent_1)
    x_0_mean_1 = x_0_1.mean(axis=1)
    x_0_2 = np.random.rand(dim, n_agent_2)
    x_0_mean_2 = x_0_2.mean(axis=1)
    x_0_3 = np.random.rand(dim, n_agent_3)
    x_0_mean_3 = x_0_3.mean(axis=1)

    eta_2_1 = 2 / (p_1.L + p_1.sigma)
    eta_1_1 = 1 / p_1.L
    eta_2_2 = 2 / (p_2.L + p_2.sigma)
    eta_1_2 = 1 / p_2.L
    eta_2_3 = 2 / (p_3.L + p_3.sigma)
    eta_1_3 = 1 / p_3.L

    n_dgd_iters = n_iters * 100
    distributed5 = [

        Q_DSAGA_tracking(p_1, n_iters=n_dgd_iters, eta=0.02, x_0=x_0_1, W=W_1, n_agent=n_agent_1, m=m_1, dim=dim,
                               s=8, name='Q-SGT, ring graph'),
        Q_DSAGA_tracking(p_2, n_iters=n_dgd_iters, eta=0.02, x_0=x_0_2, W=W_2, n_agent=n_agent_2, m=m_2, dim=dim,
                               s=8, name='Q-SGT, star graph'),
        Q_DSAGA_tracking(p_3, n_iters=n_dgd_iters, eta=0.02, x_0=x_0_3, W=W_3, n_agent=n_agent_3,
                               m=m_3, dim=dim, name='Q-SGT, E-R graph')

    ]

    res = run_exp(distributed5, kappa=kappa, max_iter=n_iters, name='logistic_regression', n_cpu_processes=4, save=True)


    plt.show()
