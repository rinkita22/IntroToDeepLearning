"""SegmentationNN"""
import torch
import torch.nn as nn


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        # conv1
        kernel_size = 3
        channels = 3
        num_filters = 32
        padding = int((kernel_size-1)/2)
        self.conv1_1 = nn.Conv2d(channels, num_filters, kernel_size, padding=padding,stride=1)
        self.conv1_1.weight.data.mul_(0.001)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size, padding=padding,stride=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 1/2
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size, padding=padding,stride=1)       
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 256 , kernel_size, padding=padding,stride=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 1/2
        
        self.upscore1 = nn.ConvTranspose2d(
           256 , num_classes, kernel_size = 4, stride=2,padding =1, bias=False)
        self.upscore2 = nn.ConvTranspose2d(
           num_classes, num_classes, kernel_size = 4, stride=2,padding =1, bias=False)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h) 
        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h) 
        h = self.upscore1(h)
        h = self.upscore2(h)
        

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return h

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
