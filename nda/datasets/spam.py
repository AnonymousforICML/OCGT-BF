#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
from nda.datasets import Dataset
from nda import log

class SpamEmail(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
        self.local_path = os.path.join(self.data_dir, 'spambase.data')
        
    def load_raw(self):
        """加载并预处理Spam Email数据集"""
        # 下载数据集
        if not os.path.exists(self.local_path):
            self.download_file(self.data_url, self.local_path)
            
        # 加载数据
        data = np.loadtxt(self.local_path, delimiter=',')
        
        # 分离特征和标签
        X = data[:, :-1]  # 所有特征列
        y = data[:, -1]   # 最后一列是标签
        
        # 划分训练集和测试集 (80% 训练, 20% 测试)
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        self.X_train = X[train_idx]
        self.Y_train = y[train_idx]
        self.X_test = X[test_idx]
        self.Y_test = y[test_idx]
        
    def normalize_data(self):
        """标准化数据"""
        # 对每个特征进行最大最小值归一化
        feature_max = np.max(self.X_train, axis=0)
        feature_min = np.min(self.X_train, axis=0)
        
        # 避免除零
        scale = feature_max - feature_min
        scale[scale == 0] = 1
        
        self.X_train = (self.X_train - feature_min) / scale
        self.X_test = (self.X_test - feature_min) / scale 