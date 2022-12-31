import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # formula to calculate size = (n+2P-f)//s +1; 
        # 'n' is the input size (h/w), 'P' is the padding, 'f' is the kernel size, 's' is the stride
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.apply(self._init_weights)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.6)
        
#         self.bn = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(13*13*256, 512) # 13*13*256=43,264
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 136)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ''' Architecture:
            - Conv1                 | input size: 224x224x1 (given image will be h=w=224pxl)  output size: 220x220x32
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
        x = self.dropout1(self.pool(F.elu(self.conv1(x))))
        x = self.dropout2(self.pool(F.elu(self.conv2(x))))
        x = self.dropout3(self.pool(F.elu(self.conv3(x))))
        x = self.dropout4(self.pool(F.elu(self.conv4(x))))
        # flatten the array
        x = x.view(x.size(0), -1) # torch.flatten(x,1)
        
        x = self.dropout5(F.elu(self.fc1(x))) 
        x = self.dropout6(F.elu(self.fc2(x)))
        x = self.fc3(x)

        # x = F.log_softmax(self.fc2(x)) # Check about the param of softmax        
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
