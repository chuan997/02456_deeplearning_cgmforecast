import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor

class Truncate(nn.Module):
    def __init__(self, length):
        super(Truncate, self).__init__()
        self.length = length

    def forward(self, x):
        return x[:, :, :-self.length].contiguous()


class CausalConv(nn.Module):
    
    def __init__(self, 
                 num_inputs=4, 
                 dilations=[1,2,4], 
                 h1=32, 
                 h2=64):
    
        super(CausalConv, self).__init__()
        self.num_inputs = num_inputs
        self.hidden_units = [h1, h1, h2]
        self.dilations = dilations

        #padding=（kernel_size-1）*dilation
        self.conv1 = nn.Conv1d(in_channels=num_inputs, 
                               out_channels=self.hidden_units[0], 
                               kernel_size=2, 
                               padding=(2-1)*self.dilations[0], 
                               dilation=self.dilations[0])
        
        self.truncate1 = Truncate((2-1)*self.dilations[0])
        #self.bn1 = nn.BatchNorm1d(self.hidden_units[0])
        #self.dropout1 = nn.Dropout()
        
        self.conv2 = nn.Conv1d(in_channels=self.hidden_units[0], 
                               out_channels=self.hidden_units[1], 
                               kernel_size=2, 
                               padding=(2-1)*self.dilations[1], 
                               dilation=self.dilations[1])
        
        self.truncate2 = Truncate((2-1)*self.dilations[1])
        
        self.conv3 = nn.Conv1d(in_channels=self.hidden_units[1], 
                               out_channels=self.hidden_units[2], 
                               kernel_size=2, 
                               padding=(2-1)*self.dilations[2], 
                               dilation=self.dilations[2])
        
        self.truncate3 = Truncate((2-1)*self.dilations[2])
        
        self.relu = nn.ReLU()
        

        self.net = nn.Sequential(self.conv1, 
                                 self.truncate1,
                                 self.relu, 
                                 self.conv2, 
                                 self.truncate2,
                                 self.relu, 
                                 self.conv3,
                                 self.truncate3)
        
    def forward(self, x):

        output = self.net(x)

        return output


class DilatedConv(nn.Module):

    def __init__(self, 
                 num_inputs=64, 
                 dilations=[1,2,4], 
                 h1=32, 
                 h2=16):
    
        super(DilatedConv, self).__init__()
        self.num_inputs = num_inputs
        self.hidden_units = [h1, h2, h2]
        self.dilations = dilations

        self.conv1 = nn.Conv1d(in_channels=num_inputs, 
                               out_channels=self.hidden_units[0], 
                               kernel_size=2,  
                               dilation=self.dilations[0])
        
        #self.truncate1 = Truncate((3-1)*self.dilations[0])
        #self.bn1 = nn.BatchNorm1d(self.hidden_units[0])
        #self.dropout1 = nn.Dropout()
        
        self.conv2 = nn.Conv1d(in_channels=self.hidden_units[0], 
                               out_channels=self.hidden_units[1], 
                               kernel_size=2,  
                               dilation=self.dilations[1])
        
        
        self.conv3 = nn.Conv1d(in_channels=self.hidden_units[1], 
                               out_channels=self.hidden_units[2], 
                               kernel_size=2, 
                               dilation=self.dilations[2])
        
        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.conv1, 
                                 self.relu, 
                                 self.conv2, 
                                 self.relu, 
                                 self.conv3)

    def forward(self, x):

        output = self.net(x)

        return output


class PostProcess(nn.Module):

    def __init__(self, num_inputs=64, h=32):
        
        super(PostProcess, self).__init__()

        self.num_inputs = num_inputs
        self.hidden_units = h

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=self.num_inputs, 
                               out_channels=self.hidden_units, 
                               kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_units, 
                               out_channels=1, 
                               kernel_size=1)
        
        self.net = nn.Sequential(self.relu,
                                 self.conv1,
                                 self.relu,
                                 self.conv2)
    
    def forward(self, x):

        output = self.net(x)
        # print('ourputshape:{}'.format(output.shape))
        return output[:,:,-1]

        

class GluNet(nn.Module):

    def __init__(self, 
                 layers=1,
                 n_steps_past=16, 
                 num_inputs=16): #[32,32,32,64,64]):
            
        super(GluNet, self).__init__()

        #self.file_name = os.path.basename(__file__)
        self.layers = layers
        #self.hidden_units = [h1,h1,h1,h2,h2]
        #self.dilations = dilations
        self.num_inputs = num_inputs
        #self.receptive_field = sum(dilations) + 1
        self.input_width = n_steps_past # n steps past
        
        self.causal = CausalConv()
        
        # in k layers
        self.dilated = DilatedConv()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv1d(in_channels=num_inputs, 
                               out_channels=64, 
                               kernel_size=1)
        
        self.postprocess = PostProcess()


    def forward(self, x):
        
        #print('=====start=====')
        #print(x.shape)
        residual = self.causal(x)
        #print(residual.shape)
        output = self.dilated(residual)
        #print(output.shape)
        filters = self.tanh(output)
        #print(filters.shape)
        gates = self.sigmoid(output)
        #print(gates.shape)
        

        x = filters * gates
        #print(x.shape)
        skip = self.conv1(x)
        #print(skip.shape)
        x = self.postprocess(skip)
        #print(x.shape)

        return x

        




# lable transform and recover
        



