# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:23:07 2020

@author: abennis
"""
import torch


def total_loss_gpu(model, X, Y, device, regularization_parameter):
    outputs = model(X.to(device))
    nll = first_operand_of_total_loss(outputs, Y.to(device))
    penalty = penalty_term(model, regularization_parameter)
    return nll + penalty, nll


def total_loss(model, inputs, targets, regularization_parameter):
    outputs = model(inputs)
    nll = first_operand_of_total_loss(outputs, targets)
    penalty = penalty_term(model, regularization_parameter)
    return nll + penalty, nll


def penalty_term(model, regularization_parameter):
    with torch.enable_grad():
        param = torch.abs(model.mwsl.weight)
        penalty = regularization_parameter * (torch.sum(torch.sqrt(param)))

    return penalty.squeeze(0)


def negative_loglikelihood_of_mixture_of_weibulls(times, deltas, alphas, betas, etas):
    t_over_eta = torch.div(times, etas)
    h1 = torch.exp(-torch.pow(t_over_eta, betas))
    h0 = torch.exp(-torch.pow(t_over_eta, betas))
    h1bis = torch.pow(t_over_eta, betas - 1)
    c = torch.div(torch.mul(alphas, betas), etas)
    return -torch.mean(deltas * torch.log(torch.sum(torch.mul(torch.mul(c, h1bis), h1), 0))
                       + (1 - deltas) * torch.log(torch.sum(alphas * h0, 0)))


def first_operand_of_total_loss(output, y):
    """
       Returns the negative log likelihood of the mixture of Weibull distributions.
       This is the first operand of the loss proposed for the network.

       Arguments:
            output {list}  : list of triplets (beta_k, eta_k, alpha_k)_{k=1,..,p_max}
            y      {tensor}: contains two columns: times recorded and event indicator
    """

    times = y[:, 0]
    deltas = y[:, 1]
    alphas = (output[2]).t()
    etas = (output[1]).t() + 1 + 1e-4
    betas = (output[0]).t() + 2
    return negative_loglikelihood_of_mixture_of_weibulls(times, deltas, alphas, betas, etas)
