# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:00:08 2020

@author: abennis
"""

from torch.utils import data

import numpy as np

import progressbar
from utils.loss import total_loss_gpu, total_loss


def train_network_with_gpu(model, X, Y, n_epochs, optimizer, regularization_parameter, device):
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss, nll = total_loss_gpu(model, X, Y, device, regularization_parameter)
        loss.backward()
        optimizer.step()

        if epoch % (n_epochs // 10) == 0 and ~np.isnan(nll.item()):
            print(' iteration n°  %d negative log-likelihood: %.8f' % (epoch + 1, nll.item()))
        model.mwsl.normalize_weights()


def train_network_without_gpu(model, trainloader, n_epochs, optimizer, regularization_parameter):
    bar = progressbar.ProgressBar(maxval=n_epochs,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    iterloader = iter(trainloader)
    for epoch in range(n_epochs):
        try:
            batch = next(iterloader)
        except StopIteration:
            iterloader = iter(trainloader)
            batch = next(iterloader)
        inputs = batch[:, :-2]
        targets = batch[:, -2:]
        optimizer.zero_grad()
        loss, nll = total_loss(model, inputs, targets, regularization_parameter)
        loss.backward()
        optimizer.step()
        bar.update(epoch)
    bar.finish()


def train_network_and_return_outputs(model, X, Y, train_inds, xfold_train, yfold_train,
                                     xfold_val, optimizer, n_epochs, regularization_parameter,
                                     gpu, device):
    if gpu:
        model = model.to(device)

        train_network_with_gpu(model, xfold_train, yfold_train, n_epochs,
                               optimizer, regularization_parameter, device)

        outputs = model.cpu()(xfold_val)

    else:

        trainloader = data.DataLoader(np.concatenate((X, Y), axis=1),
                                      batch_size=16, shuffle=False,
                                      sampler=data.sampler.SubsetRandomSampler(train_inds), num_workers=4)

        train_network_without_gpu(model, trainloader, n_epochs, optimizer, regularization_parameter)
        outputs = model(xfold_val)

    return outputs
