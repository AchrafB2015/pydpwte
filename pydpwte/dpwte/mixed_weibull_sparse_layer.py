# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:13:57 2020

@author: abennis
"""
import torch
import torch.nn as nn


class SparseWeibullMixtureLayer(nn.Module):
    """ Class that creates the Sparse Weibull Mixture layer
        Arguments:
            p_max:  the upper bound of the number of Weibull distributions composing the mixture
    """

    def __init__(self, p_max):
        super().__init__()
        self.p_max = p_max
        self.weight = torch.nn.Parameter(torch.Tensor(p_max))
        self.initialize_weights()

    def initialize_weights(self):
        """ Initializes the MWS's weights following
        the uniform distribution of support [0,1] """

        torch.nn.init.uniform_(self.weight, 0, 1)

    def normalize_weights(self):
        self.weight = nn.Parameter(self.weight / self.weight.sum())

    def forward(self, alphas):
        """ Function that calculates element-wise multiplication
            wbar x alpha = [w_bar_1*alpha_1, ..., w_bar_p*alpha_p]
            where w_bar = w/sum(w)
            alphas {tensor}: the outputs of the layer that precedes SWM layer.
        """

        x, y = alphas.shape
        if y != self.p_max:
            print(f'Wrong Input Features. Please use tensor with {self.p_max} Input Features')
            return 0
        w_bar = self.weight / (self.weight.sum())
        return alphas * w_bar

    def extra_repr(self):
        return 'p_max={}'.format(
            self.p_max is not None
        )
