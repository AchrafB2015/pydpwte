# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:20:30 2020

@author: abennis
"""
import numpy as np


def normalize_input_data_standard_mode(x):
    nb_features = x.shape[1]
    for j in range(nb_features):
        if np.std(x[:, j]) != 0:
            x[:, j] = (x[:, j] - np.mean(x[:, j])) / np.std(x[:, j])
        else:
            x[:, j] = x[:, j] - np.mean(x[:, j])
    return x


def normalize_input_data_normal_mode(x):
    nb_features = x.shape[1]
    for j in range(nb_features):
        x[:, j] = (x[:, j] - np.min(x[:, j])) / (np.max(x[:, j]) - np.min(x[:, j]))
    return x


def normalize_input_data(x, norm_mode='standard'):
    if norm_mode == 'standard':
        x = normalize_input_data_standard_mode(x)
    elif norm_mode == 'normal':
        x = normalize_input_data_normal_mode(x)
    else:
        x = None
    return x
