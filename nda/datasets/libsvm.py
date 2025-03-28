#!/usr/bin/env python
# coding=utf-8
import os
from urllib import request
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

def download_if_not_exists(url, filepath):
    """下载文件如果不存在"""
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        request.urlretrieve(url, filepath)

def _load_raw(name, n_features=None):
    """加载原始数据"""
    data_dir = os.path.expanduser(f'~/data/LibSVM/{name}')
    data_path = os.path.join(data_dir, name)
    
    if not os.path.exists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}"
        download_if_not_exists(url, data_path)
    
    X, Y = load_svmlight_file(data_path, n_features=n_features)
    return X, Y

class LibSVM:
    def __init__(self, name, normalize=True):
        self.name = name
        self.normalize = normalize
        
    def load(self):
        # 加载训练集和测试集
        X_train, Y_train = _load_raw(self.name)
        test_name = f"{self.name}.t"
        
        try:
            X_test, Y_test = _load_raw(test_name)
        except:
            X_test, Y_test = X_train, Y_train
        
        # 确保特征数量一致
        max_features = max(X_train.shape[1], X_test.shape[1])
        
        # 如果需要，扩展特征矩阵
        if X_train.shape[1] < max_features:
            X_train = self._pad_features(X_train, max_features)
        if X_test.shape[1] < max_features:
            X_test = self._pad_features(X_test, max_features)
            
        # 转换为密集矩阵
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        
        # 标准化数据
        if self.normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        return X_train, Y_train, X_test, Y_test
    
    def _pad_features(self, X, n_features):
        """将特征矩阵扩展到指定的特征数"""
        if X.shape[1] >= n_features:
            return X
            
        # 创建新的特征矩阵
        X_padded = csr_matrix((X.shape[0], n_features))
        X_padded[:, :X.shape[1]] = X
        return X_padded

# 确保类被正确定义和导出
