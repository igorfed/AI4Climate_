# Imports
import torch
from torch import nn  # All neural network modules
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions

class CNN(nn.Module):

    def __init__(self, in_channels=1, num_classes = 10):
        # call the parent constructor
        super(CNN, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(
            in_channels=1, # it is monochrome (channels of the images)
            out_channels = 8,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8, # it is monochrome (channels of the images)
            out_channels = 16,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        # CONV => RELU => POOL layer 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # CONV => RELU => POOL layer 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
