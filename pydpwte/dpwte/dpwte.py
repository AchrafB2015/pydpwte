# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:15:08 2020

@author: abennis
"""
import torch
import torch.nn as nn
import torch.nn.functional as t_func

from dpwte.mixed_weibull_sparse_layer import SparseWeibullMixtureLayer


class dpwte(nn.Module):
    
    """
        Class for DPWTE Network;
        Arguments:
            n_cols {int}        : number of features of the input.
            p_max  {int}        : upper bound of the mixture size
            sparse_reg {boolean}: Use the 'Weibull Sparse Mixture' if True.
            lambda_reg {float}  : regularization parameter in the loss function used to train DPWTE.
    """

    def __init__(self, n_cols, p_max, sparse_reg=False, lambda_reg=1e-4):
        super(dpwte, self).__init__()

        # Layers in the Shared Sub-Network (SSN)
        self.dense_1_SSN = nn.Linear(n_cols, 128)
        self.batch_SSN   = nn.BatchNorm1d(128)
        self.dense_2_SSN = nn.Linear(128, 64)
        self.dense_3_SSN = nn.Linear(64, 32)

        # Layers in the Regression Sub-Network (RSN)
        self.dense_1_RSN     = nn.Linear(32, 16)
        self.dense_2_RSN     = nn.Linear(16, 8)
        self.batch_RSN       = nn.BatchNorm1d(8)
        self.betasout         = nn.Linear(8, p_max)
        self.etasout          = nn.Linear(8, p_max)

        # Layers in the Classifier Sub-Network (CSN)
        self.dense_1_CSN       = nn.Linear(32, 16)
        self.dense_2_CSN       = nn.Linear(16, 8)
        self.batch_CSN         = nn.BatchNorm1d(8)
        self.alphasout = nn.Linear(8, p_max)
        
        self.mwsl              = SparseWeibullMixtureLayer(p_max)
        self.walpha            = nn.Parameter(torch.randn(p_max), requires_grad=True)
        self.sparse_reg        = sparse_reg
        self.lambda_reg        = lambda_reg

    def forward(self, x):
        # shared sub-network
        x_prime  = t_func.relu(self.batch_SSN(self.dense_1_SSN(x)))
        x_second = t_func.relu((self.dense_2_SSN(x_prime)))
        z        = t_func.relu((self.dense_3_SSN(x_second)))

        # regression sub-network
        x_reg      = t_func.relu(self.dense_1_RSN(z))
        x_reg      = t_func.relu(self.batch_RSN(self.dense_2_RSN(x_reg)))
        betas      = t_func.elu(self.betasout(x_reg))
        etas       = t_func.elu(self.etasout(x_reg))

        # classifier sub-network
        x_clf = t_func.relu(self.dense_1_CSN(z))
        x_clf = t_func.relu(self.batch_CSN(self.dense_2_CSN(x_clf)))
        
        if (self.sparse_reg):# Mixed Weibull Sparse layer
            q_k            = t_func.softmax(self.alphasout(x_clf))
            alphas         = self.mwsl(q_k)
            sums_of_alphas = alphas.data.sum(dim=1).reshape(-1, 1)# sums_of_alphas = (alphas.data).sum()
            alphas         = alphas / sums_of_alphas
        else:
            alphas  = t_func.softmax(self.alphasout(x_clf))
        
        
        return [betas, etas, alphas]
