## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # formula to calculate size = (n+2P-f)//s +1; 
        # 'n' is the input size (h/w), 'P' is the padding, 'f' is the kernel size, 's' is the stride
        self.conv2 = nn.Conv2d(32, 64, 5) 
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
        self.bn = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(25*25*128, 128) # 25*25*128=80,000
        self.fc2 = nn.Linear(128, 136)
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ''' Architecture:
            - Conv1                 | input size: 224x224x1 (given image will be h=w=224pxl)  output size: 220x220x32
            - Batch normalization   | input size: 220x220x32; output size: 220x220x32
            - activation (relu)     | input size: 220x220x32; output size: 220x220x32
            - max pool              | input size: 220x220x32; output size: 110x110x32
            
            - conv2                 | input size: 110x110x32; output size: 106x106x64
            - drop-out1             | input size: 106x106x64; output size: 106x106x64
            - activation(relu)      | input size: 106x106x64; output size: 106x106x64
            - max pool              | input size: 106x106x64; output size: 53x53x64
            
            - conv3                 | input size: 53x53x64; output size: 51x51x128
            - activation (relu)     | input size: 53x53x64; output size: 51x51x128
            - max pool              | input size: 53x53x64; output size: 25x25x128
            
            - fully connected 1     | input size: 80,000x1; output size: 128x1
            - drop-out2             | input size: 128x1; output size: 128x1
            - activation (relu)     | input size: 128x1; output size: 128x1
            - fully connected 2     | input size: 128x1; output size: 136x1
            - softmax               | input size: 136x1; output size: 136x1
        '''
        x = self.pool(F.relu(self.bn(self.conv1(x))))
        x = self.pool(F.relu(self.dropout1(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
            # flatten the array
        x = x.view(x.size(0), -1) # torch.flatten(x,1)
        x = F.relu(self.dropout2(self.fc1(x))) # check the input size
        x = self.fc2(x)
#         x = F.log_softmax(self.fc2(x)) # Check about the param of softmax        
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
