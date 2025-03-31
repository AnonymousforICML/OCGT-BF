#!/usr/bin/env python
# coding=utf-8
import numpy as np


import networkx as nx
import matplotlib.pyplot as plt
from nda import log


class Problem(object):
    '''The base problem class, which generates the random problem and supports function value and gradient evaluation'''
# 初始化
    def __init__(self, n_agent=20, m=1000, dim=40, graph_type='er', graph_params=None, regularization=None, r=0.3, dataset ='random', sort=False, shuffle=False, normalize_data=True, gpu=False):
    # 初始化参数
        self.n_agent = n_agent          # Number of agents
        self.m = m                      # Number of samples per agent
        self.dim = dim                  # Dimension of the variable
        self.X_total = None             # All data
        self.Y_total = None             # All labels
        self.X = []                     # Distributed data
        self.Y = []                     # Distributed labels
        self.x_0 = None                 # The true varibal value
        self.x_min = None               # The minimizer varibal value
        self.f_min = None               # The optimal function value
        self.L = None                   # The smoothness constant
        self.sigma = 0                  # The strong convexity constant
        self.is_smooth = True           # If the problem is smooth or not
        self.r = r
        self.graph_params = graph_params
        self.graph_type = graph_type
        self.dataset = dataset

    # 初始化数据集
        # 随机生成数据集

        from nda import datasets


        self.X_train, self.Y_train, self.X_test, self.Y_test = datasets.LibSVM(
            name=dataset, 
            normalize=normalize_data
        )


        self.X_train = np.append(self.X_train, np.ones((self.X_train.shape[0], 1)), axis=1)
        self.X_test = np.append(self.X_test, np.ones((self.X_test.shape[0], 1)), axis=1)
        # 数据处理
        self.m = self.X_train.shape[0] // n_agent   # 整除
        self.m_total = self.m * n_agent

        self.X_train = self.X_train[:self.m_total]
        self.Y_train = self.Y_train[:self.m_total]
        self.dim = self.X_train.shape[1]
        print(self.X_train.shape)

        print(self.Y_train.shape)



    # 数据集拆分
        self.X = self.split_data(self.X_train)
        self.Y = self.split_data(self.Y_train)
    # 生成连通图
        self.generate_graph(graph_type=graph_type, params=graph_params)
        self.plot_graph()
    # 正则化参数
        if regularization == 'l1':
            self.grad_g = self._grad_regularization_l1
            self.is_smooth = False

        elif regularization == 'l2':
            self.grad_g = self._grad_regularization_l2

    def cuda(self):
        log.debug("Copying data to GPU")

        # Copy every np.array to GPU if needed
        for k in self.__dict__:
            if type(self.__dict__[k]) == np.ndarray:
                self.__dict__[k] = np.array(self.__dict__[k])
# 拆分数据
    def split_data(self, X):
        '''Helper function to split data according to the number of training samples per agent.'''
        if self.m * self.n_agent != len(X):
            log.fatal('Data cannot be distributed equally to %d agents' % self.n_agent)
        if X.ndim == 1:
            return X.reshape(self.n_agent, -1)
        else:
            return X.reshape(self.n_agent, self.m, -1)

# 梯度计算 f(x)=h(x)+g(x)
    def grad(self, w, i=None, j=None):
        '''(sub-)Gradient of f(x) = h(x) + g(x) at w. Depending on the shape of w and parameters i and j, this function behaves differently:
        1. If w is a vector of shape (dim,)
            1.1 If i is None and j is None
                returns the full gradient.
            1.2 If i is not None and j is None
                returns the gradient at the i-th agent.
            1.3 If i is None and j is not None
                returns the i-th gradient of all training data.
            1.4 If i is not None and j is not None
                returns the gradient of the j-th data sample at the i-th agent.
            Note i, j can be integers, lists or vectors.
        2. If w is a matrix of shape (dim, n_agent)
            2.1 if j is None
                returns the gradient of each parameter at the corresponding agent
            2.2 if j is not None
                returns the gradient of each parameter of the j-th sample at the corresponding agent.
            Note j can be lists of lists or vectors.
        '''
        return self.grad_h(w, i=i, j=j) + self.grad_g(w)

    def grad_h(self, w, i=None, j=None):
        '''Gradient of h(x) at w. Depending on the shape of w and parameters i and j, this function behaves differently:
        1. If w is a vector of shape (dim,)
            1.1 If i is None and j is None
                returns the full gradient.
            1.2 If i is not None and j is None
                returns the gradient at the i-th agent.
            1.3 If i is None and j is not None
                returns the i-th gradient of all training data.
            1.4 If i is not None and j is not None
                returns the gradient of the j-th data sample at the i-th agent.
            Note i, j can be integers, lists or vectors.
        2. If w is a matrix of shape (dim, n_agent)
            2.1 if j is None
                returns the gradient of each parameter at the corresponding agent
            2.2 if j is not None
                returns the gradient of each parameter of the j-th sample at the corresponding agent.
            Note j can be lists of lists or vectors.
        '''
        pass

    def grad_g(self, w):
        '''Sub-gradient of g(x) at w. Returns the sub-gradient of corresponding parameters. w can be a vector of shape (dim,) or a matrix of shape (dim, n_agent).
        '''
        return 0

    def f(self, w, i=None, j=None, split='train'):
        '''Function value of f(x) = h(x) + g(x) at w. If i is None, returns the global function value; if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''
        return self.h(w, i=i, j=j, split=split) + self.g(w)

    def hessian(self, *args, **kwargs):
        raise NotImplementedError

    def h(self, w, i=None, j=None, split='train'):
        '''Function value at w. If i is None, returns h(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''
        raise NotImplementedError

    def g(self, w):
        '''Function value of g(x) at w. Returns 0 if no regularization.'''
        return 0
    
# 正则化项
    def _regularization_l1(self, w):
        return self.r * np.abs(w).sum(axis=0)

    def _regularization_l2(self, w):
        return self.r * (w * w).sum(axis=0)
    
# 正则化项的梯度
    def _grad_regularization_l1(self, w):
        g = np.zeros(w.shape)
        g[w > 1e-5] = 1
        g[w < -1e-5] = -1
        return self.r * g

    def _grad_regularization_l2(self, w):
        return 2 * self.r * w

# 梯度正确性验证
    def grad_check(self):
        '''Check whether the full gradient equals to the gradient computed by finite difference at a random point.'''
        w = np.random.randn(self.dim)
        delta = np.zeros(self.dim)
        grad = np.zeros(self.dim)
        eps = 1e-4

        for i in range(self.dim):
            delta[i] = eps
            grad[i] = (self.f(w + delta) - self.f(w - delta)) / 2 / eps
            delta[i] = 0

        error = np.linalg.norm(grad - self.grad(w))
        if error > eps:
            log.warn('Gradient implementation check failed with difference %.4f!' % error)
            return False
        else:
            log.info('Gradient implementation check succeeded!')
            return True

 # 检查分布式环境下函数值和梯度计算是否正确
    def distributed_check(self):
        '''Check the distributed function and gradient implementations are correct.'''

        # 检查一维情况下梯度的计算是否正确
        def _check_1d_gradient():
            '''计算全局梯度，各个代理的局部梯度以及所有样本的梯度，并比较它们之间的差异。'''
            w = np.random.randn(self.dim)
            g = self.grad(w)
            g_i = g_ij = 0
            res = True

            for i in range(self.n_agent):
                _tmp_g_i = self.grad(w, i)
                _tmp_g_ij = 0
                for j in range(self.m):
                    _tmp_g_ij += self.grad(w, i, j)

                if np.linalg.norm(_tmp_g_i - _tmp_g_ij / self.m) > 1e-5:
                    log.warn('Distributed graident check failed! Difference between local graident at agent %d and average of all local sample gradients is %.4f' % (i, np.linalg.norm(_tmp_g_i - _tmp_g_ij / self.m)))
                    res = False

                g_i += _tmp_g_i
                g_ij += _tmp_g_ij

            g_i /= self.n_agent
            g_ij /= self.m_total

            if np.linalg.norm(g - g_i) > 1e-5:
                log.warn('Distributed gradient check failed! Difference between global graident and average of local gradients is %.4f', np.linalg.norm(g - g_i))
                res = False

            if np.linalg.norm(g - g_ij) > 1e-5:
                log.warn('Distributed graident check failed! Difference between global graident and average of all sample gradients is %.4f' % np.linalg.norm(g - g_ij))
                res = False

            return res

        # 检查二维情况下梯度的计算是否正确
        def _check_2d_gradient():
        
            res = True
            w_2d = np.random.randn(self.dim, self.n_agent)

            g_1d = 0
            for i in range(self.n_agent):
                g_1d += self.grad(w_2d[:, i], i=i)

            g_1d /= self.n_agent
            g_2d = self.grad(w_2d).mean(axis=1)

            if np.linalg.norm(g_1d - g_2d) > 1e-5:
                log.warn('Distributed graident check failed! Difference between global gradient and average of distributed graidents is %.4f' % np.linalg.norm(g_1d - g_2d))
                res = False

            g_2d_sample = self.grad(w_2d, j=np.arange(self.m).reshape(-1, 1).repeat(self.n_agent, axis=1).T).mean(axis=1)

            if np.linalg.norm(g_1d - g_2d_sample) > 1e-5:
                log.warn('Distributed graident check failed! Difference between global graident and average of all sample gradients is %.4f' % np.linalg.norm(g_1d - g_2d_sample))
                res = False

            samples = np.random.randint(0, self.m, (self.n_agent, 10))
            g_2d_stochastic = self.grad(w_2d, j=samples)
            for i in range(self.n_agent):
                g_1d_stochastic = self.grad(w_2d[:, i], i=i, j=samples[i])
                if np.linalg.norm(g_1d_stochastic - g_2d_stochastic[:, i]) > 1e-5:
                    log.warn('Distributed graident check failed! Difference between distributed stoachastic gradient at agent %d and average of sample gradients is %.4f' % (i, np.linalg.norm(g_1d_stochastic - g_2d_stochastic[:, i])))
                    res = False

            return res
        
        # 检查函数值的计算是否正确
        '''通过比较全局函数值、各个代理的局部函数值以及所有样本的函数值之间的差异来验证实现'''
        def _check_function_value():
            w = np.random.randn(self.dim)
            f = self.f(w)
            f_i = f_ij = 0
            res = True

            for i in range(self.n_agent):
                _tmp_f_i = self.f(w, i)
                _tmp_f_ij = 0
                for j in range(self.m):
                    _tmp_f_ij += self.f(w, i, j)

                if np.abs(_tmp_f_i - _tmp_f_ij / self.m) > 1e-10:
                    log.warn('Distributed function value check failed! Difference between local function value at agent %d and average of all local sample function values %d is %.4f' % (i, i, np.abs(_tmp_f_i - _tmp_f_ij / self.m)))
                    res = False

                f_i += _tmp_f_i
                f_ij += _tmp_f_ij

            f_i /= self.n_agent
            f_ij /= self.m_total

            if np.abs(f - f_i) > 1e-10:
                log.warn('Distributed function value check failed! Difference between the global function value and average of local function values is %.4f' % np.abs(f - f_i))
                res = False

            if np.abs(f - f_ij) > 1e-10:
                log.warn('Distributed function value check failed! Difference between the global function value and average of all sample function values is %.4f' % np.abs(f - f_ij))
                res = False

            return res

        res = _check_function_value() & _check_1d_gradient() & _check_2d_gradient()
        if res:
            log.info('Distributed check succeeded!')
            return True
        else:
            return False

# 生成连通图

    def generate_graph(self, graph_type='expander', params=None):
        '''Generate connected connectivity graph according to the params.'''

        # Set random seed
        random_seed = 42

        if graph_type == 'expander':
            G = nx.paley_graph(self.n_agent).to_undirected()
        elif graph_type == 'grid':
            G = nx.grid_2d_graph(*params)
        elif graph_type == 'cycle':
            G = nx.cycle_graph(self.n_agent)
        elif graph_type == 'path':
            G = nx.path_graph(self.n_agent)
        elif graph_type == 'star':
            G = nx.star_graph(self.n_agent - 1)
        elif graph_type == 'er':
            if params < 2 / (self.n_agent - 1):
                log.fatal("Need higher probability to create a connected E-R graph!")
            G = None
            while G is None or nx.is_connected(G) is False:
                G = nx.erdos_renyi_graph(self.n_agent, params, seed=random_seed)
        elif graph_type in ['scale-free', 'scale_free']:
            # For scale-free network (Barabási-Albert model)
            # params should be the number of edges to attach from a new node to existing nodes
            # Default is 1 if not specified
            m = params if params is not None else 1

            # Ensure the network is connected and has the correct number of nodes
            G = None
            while G is None or nx.is_connected(G) is False or G.number_of_nodes() < self.n_agent:
                G = nx.barabasi_albert_graph(self.n_agent, m, seed=random_seed)
        else:
            log.fatal('Graph type %s not supported' % graph_type)

        self.n_edges = G.number_of_edges()
        self.G = G

# 画出这个图
    def plot_graph(self):
        '''Plot the generated connectivity graph.'''

        node_style = {
            'node_color': 'blue',  # 节点颜色
            'node_size': 30,  # 节点大小
            'node_shape': 'o',  # 节点形状
            'alpha': 0.5  # 透明度
        }
        edge_style = {
            'edge_color': 'gray',  # 边的颜色
            'width': 1  # 边的宽度
        }
        plt.figure()
        #nx.draw(self .G)
        nx.draw(self.G, **node_style, **edge_style)
        plt.savefig(self.graph_type + '_graph.eps', format='eps')
