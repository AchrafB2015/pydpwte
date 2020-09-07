# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:15:08 2020

@author: abennis
"""
import torch
import torch.nn as nn
import torch.nn.functional as t_func

from dpwte.mixed_weibull_sparse_layer import SparseWeibullMixtureLayer


class Dpwte(nn.Module):
    """
        Class for DPWTE model;
        Arguments:
            n_cols {int} : number of features of the input.
            p_max  {int} : upper bound of the mixture size
    """

    def __init__(self, n_cols, p_max):
        super(Dpwte, self).__init__()

        # Layers in the Shared Sub-Network (SSN)
        self.dense_1_SSN = nn.Linear(n_cols, 128)
        self.batch_SSN = nn.BatchNorm1d(128)
        self.dense_2_SSN = nn.Linear(128, 64)
        self.dense_3_SSN = nn.Linear(64, 32)

        # Layers in the Regression Sub-Network (RSN)
        self.dense_1_RSN = nn.Linear(32, 16)
        self.dense_2_RSN = nn.Linear(16, 8)
        self.batch_RSN = nn.BatchNorm1d(8)
        self.denseOutputBeta = nn.Linear(8, p_max)
        self.denseOutputEta = nn.Linear(8, p_max)

        # Layers in the Classifier Sub-Network (CSN)
        self.dense_1_CSN = nn.Linear(32, 16)
        self.dense_2_CSN = nn.Linear(16, 8)
        self.batch_CSN = nn.BatchNorm1d(8)
        self.denseOutputAlphas = nn.Linear(8, p_max)
        self.mwsl = SparseWeibullMixtureLayer(p_max)
        self.walpha = nn.Parameter(torch.randn(p_max), requires_grad=True)

    def forward(self, x):
        # shared sub-network
        x = t_func.relu(self.batch_SSN(self.dense_1_SSN(x)))
        x = t_func.relu((self.dense_2_SSN(x)))
        # z is the input of the classifier and regression sub-networks
        z = t_func.relu((self.dense_3_SSN(x)))

        # regression sub-network
        x1 = t_func.relu(self.dense_1_RSN(z))
        x1 = t_func.relu(self.batch_RSN(self.dense_2_RSN(x1)))
        betas = t_func.elu(self.denseOutputBeta(x1))
        etas = t_func.elu(self.denseOutputEta(x1))

        # classifier sub-network
        x2 = t_func.relu(self.dense_1_CSN(z))
        x2 = t_func.relu(self.batch_CSN(self.dense_2_CSN(x2)))
        x2 = t_func.softmax(self.denseOutputAlphas(x2))
        # Mixed Weibull Sparse layer
        x2 = self.mwsl(x2)
        # sums_of_alphas = (alphas.data).sum()
        sums_of_alphas = x2.data.sum(dim=1).reshape(-1, 1)
        alphas = x2 / sums_of_alphas
        return [betas, etas, alphas]
