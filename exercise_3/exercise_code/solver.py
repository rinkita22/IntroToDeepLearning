from random import shuffle
import numpy as np
import torch.nn as nn
import math
import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
    def calculate_mean(self, prediction, gTruth):
        count = 0
        for i in range(0, len(prediction)):
            if prediction[i] == gTruth[i]:
                count+=1
        return count / len(prediction)
    
    def calculate_sum(self, prediction, gTruth):
        count = 0
        for i in range(0, len(prediction)):
            if prediction[i] == gTruth[i]:
                count+=1
        return count
    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        crossE = nn.CrossEntropyLoss()
        r_loss = 0.0
        for epoch in range(num_epochs):
            for iteration, data in enumerate(train_loader, 0):
                inputs, labels = data
                optim.zero_grad()
                outputs = model(inputs)
                loss =  crossE(outputs, labels)
                loss.backward()
                optim.step()
                r_loss += loss.item()
                self.train_loss_history.append(math.log(loss))
                if iteration % 100 == 99:
                    print('Iteration %d / %d] Train loss: %0.4f' %
                          (iteration + 1, iter_per_epoch, r_loss/100))
                    r_loss = 0.0
                    
            r_loss = 0.0
            train_acc = 0.0
            for iteration, data in enumerate(train_loader, 0):
                inputs, labels = data
                optim.zero_grad()
                outputs = model(inputs)
                loss = crossE(outputs, labels)
                y_pred = torch.argmax(outputs, dim=1)
                r_loss += loss.item()
                train_acc += self.calculate_mean(y_pred, labels)
            train_acc = train_acc / len(train_loader)
            self.train_acc_history.append(math.log(train_acc))
            print('Epoch %d / %d] Train Accuracy / loss: %0.4f / %.4f' %
                              (epoch+1, num_epochs, train_acc / iter_per_epoch, r_loss/iter_per_epoch))

          
        
            r_loss = 0.0
            val_acc = 0.0
            for iteration, data in enumerate(val_loader, 0):
                inputs, labels = data
                optim.zero_grad()
                outputs = model(inputs)
                loss = crossE(outputs, labels)
                y_pred = torch.argmax(outputs, dim=1)
                r_loss += loss.item()
                val_acc += self.calculate_mean(y_pred, labels)
            val_acc = val_acc/len(val_loader)
            self.val_acc_history.append(math.log(val_acc / iter_per_epoch))
            self.val_loss_history.append(math.log(r_loss / iter_per_epoch))
            print('Epoch %d / %d] Val Accuracy / loss: %0.4f / %.4f' %
                              (epoch+1, num_epochs, val_acc / iter_per_epoch, r_loss/iter_per_epoch))

        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
