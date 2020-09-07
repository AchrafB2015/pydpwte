# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:22:48 2020

@author: abennis
"""
import numpy as np


def select_alphas_and_normalize(alphas, w_th):
    condition = alphas > w_th
    selected_alphas = alphas[condition]
    selected_indices = condition.flatten().numpy()
    sum_of_selected_alphas = selected_alphas.sum()
    selected_alphas /= sum_of_selected_alphas
    return list(selected_alphas), selected_indices


def select_mixture_parameters(results, selected_indices):
    beta_pred = results[0] + 2
    eta_pred = results[1] + 1 + 1e-4
    alpha_pred = results[2]
    selected_beta = beta_pred[:, selected_indices].detach().numpy()
    selected_eta = eta_pred[:, selected_indices].detach().numpy()
    selected_alpha = alpha_pred[:, selected_indices].detach().numpy()
    return np.array(selected_beta), np.array(selected_eta), np.array(selected_alpha)
