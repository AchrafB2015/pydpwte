# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:22:48 2020

@author: abennis
"""
import numpy as np


def select_alphas_and_normalize(alphas, w_th):
    
    """Post-Training Steps (cf Sect 6.3 of DPWTE paper)
    This function takes the softmax outputs and the threshold (above which alpha_k is selected) and returns the 
    list of alpha_k selected as well as their respective indices.
    Argument:
        alphas {tensor}: weighting coefficients of the Weibull distributions learned by the network.
        w_th {float}   : the weight's threshold.
    """
    condition              = alphas > w_th
    selected_alphas        = alphas[condition]
    selected_indices       = condition.flatten().numpy()
    sum_of_selected_alphas = selected_alphas.sum()
    selected_alphas       /= sum_of_selected_alphas
    return list(selected_alphas), selected_indices


def select_mixture_parameters(results, selected_indices):
    
    """ Recover the real values of the learned parameters and convert them to arrays.
    Argument:
        results {list}: the set of triplets (beta_k,eta_k,alpha_k).
        selected_indices {array}: the respective indices of the selected alphas (selected Weibulls).
    """
    
    beta_pred      = results[0] + 2
    eta_pred       = results[1] + 1 + 1e-4
    alpha_pred     = results[2]
    selected_beta  = beta_pred[:, selected_indices].detach().numpy()
    selected_eta   = eta_pred[:, selected_indices].detach().numpy()
    selected_alpha = alpha_pred[:, selected_indices].detach().numpy()
    return np.array(selected_beta), np.array(selected_eta), np.array(selected_alpha)
