from random import shuffle
import numpy as np

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

        for t in range(num_epochs):
            for idx, batch in enumerate(train_loader):
                images, labels = batch
                scores, loss = self._step(model, optim, images,labels)

                self.train_loss_history.append(loss)
                if idx % log_nth == 0:
                    print('Epoch %d / %d: \tloss: %f' %
                        (t+1, num_epochs, loss))

                if idx == len(train_loader) - 1:
                    acc = self.check_accuracy( scores, labels )
                    self.train_acc_history.append(acc)
                    print('Epoch %d / %d: \tloss: %f \t training accuracy: %f' %
                            (t+1, num_epochs, loss, acc))
            best_acc = 0.
            for batch in val_loader:
                images, labels = batch
                scores = model(images)
                val_acc = self.check_accuracy(scores, labels)
                if val_acc > best_acc:
                    best_acc = val_acc
            self.val_acc_history.append(best_acc)
            print('Epoch %d / %d: \t validation accuracy: %f' %
                (t+1, num_epochs, best_acc))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

    def check_accuracy(self, scores, labels ):
        indices = scores.argmax(1)
        value = (indices == labels).sum().item()
        return value/len(labels)

    def _step(self, model, optim, images, labels ):
        optim.zero_grad() 
        scores = model(images)
        
        loss = self.loss_func( scores, labels )
        loss.backward()
        optim.step()

        return scores, loss.item()
