#!/usr/bin/env python
# coding=utf-8

from nda.datasets.dataset import Dataset
from nda.datasets.gisette import Gisette
from nda.datasets.libsvm import LibSVM
from nda.datasets.mnist import MNIST
from .spam import SpamEmail

def LibSVM(name='gisette', normalize=True):
    if name.lower() == 'gisette':
        dataset = Gisette()
    elif name.lower() == 'spam':
        dataset = SpamEmail()
    else:
        raise ValueError(f'Dataset {name} not supported')
        
    if normalize:
        dataset.load_raw()
        dataset.normalize_data()
    return dataset.X_train, dataset.Y_train, dataset.X_test, dataset.Y_test
