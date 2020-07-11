# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:51:47 2020

@author: abennis
"""
from sklearn.model_selection import KFold
import torch
import torch.optim as optim

from scipy.special import gamma
import lifelines.utils
from dpwte.postprocess import select_alphas_and_normalize, select_mixture_parameters
from dpwte.train import train_network_and_return_outputs


class CrossValidation:

    def __init__(self, model, p, inputs, targets, optimizer_name,
                 regularization_parameter, w_th, n_epochs, lr):
        self.X = inputs
        self.Y = targets
        self.model = model
        self.n_epochs = n_epochs
        self.reg_param = regularization_parameter
        self.p = p
        self.w_th = w_th
        self.lr = lr

        self.gpu = True if torch.cuda.is_available() else False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.xfold_train = None
        self.xfold_val = None
        self.yfold_train = None
        self.yfold_val = None
        self.train_inds = None
        self.val_inds = None
        self.results = None

        if optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif optimizer_name == 'Adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def five_fold_cross_validation(self):

        count = 0
        c_index = []
        kf = KFold(n_splits=5)

        for fold_indices in kf.split(self.X):

            self.train_inds, self.val_inds = fold_indices
            trained = False
            counter = 1
            while not trained:

                self.xfold_train = torch.tensor(self.X[self.train_inds])
                self.xfold_val = torch.tensor(self.X[self.val_inds])
                self.yfold_train = torch.tensor(self.Y[self.train_inds])
                self.yfold_val = torch.tensor(self.Y[self.val_inds])
                validation_target = self.yfold_val.numpy()

                self.results = train_network_and_return_outputs(self.model, self.X,
                                                                self.Y, self.train_inds,
                                                                self.xfold_train,
                                                                self.yfold_train, self.xfold_val,
                                                                self.optimizer, self.n_epochs,
                                                                self.reg_param, self.gpu, self.device, self.p)

                mean_alphas = torch.mean(self.results[2], 0)
                selected_alphas, selected_indices = select_alphas_and_normalize(mean_alphas, self.w_th)
                betas_predicted, etas_predicted, alphas = select_mixture_parameters(self.results, selected_indices)

                if len(selected_alphas) != 0:
                    print('                                                                                          ')
                    print('##########################################################################################')
                    print('                                                                                          ')
                    print(
                        '                                       p̃ = ' + str(len(selected_alphas)) + '                ')

                    mean_lifetime_predicted = (alphas * etas_predicted * gamma(1 + (1 / betas_predicted))).sum(axis=1)
                    c_index.append(lifelines.utils.concordance_index(validation_target[:, 0],
                                                                     mean_lifetime_predicted,
                                                                     validation_target[:, 1]))
                    count += 1
                    print('                                                                                           ')
                    print('                   Fold n° ' + str(count) + ', C-index of validation data: ' + str(
                        c_index[-1]))
                    print('                                                                                           ')
                    print('###########################################################################################')
                    trained = True

                else:
                    print('retry')
                    counter += 1
                    self.n_epochs = self.n_epochs // 2
                    if counter == 3:
                        trained = True

        print('Experiment terminated')
        return c_index
