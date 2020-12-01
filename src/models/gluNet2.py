import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor


class CausalConv(nn.Module):
    
    def __init__(self, 
                 num_inputs=4, 
                 h1=32, 
                 h2=64,
                 kernel_size=2):
        
        """
        :param num_input: num of initial input channels (4)
        :param dilations: 1, 2, 4, 8 ....
        :param h1: hidden units or channels
        :param h2: hidden units or channels

        :return: shape (batch_size, h2, num_step_past)
        """
    
        super(CausalConv, self).__init__()
        self.hidden_units = [h1, h1, h2, h2]

        #padding=（kernel_size-1）*dilation
        self.conv1 = nn.Conv1d(in_channels=num_inputs, 
                               out_channels=self.hidden_units[0], 
                               kernel_size=2, dilation=1)
        
        self.conv2 = nn.Conv1d(in_channels=self.hidden_units[0], 
                               out_channels=self.hidden_units[1], 
                               kernel_size=2, dilation=1)
        
        self.conv3 = nn.Conv1d(in_channels=self.hidden_units[1], 
                               out_channels=self.hidden_units[2], 
                               kernel_size=2, dilation=1)
        
        self.conv4 = nn.Conv1d(in_channels=self.hidden_units[2], 
                               out_channels=self.hidden_units[3], 
                               kernel_size=2, dilation=1)
        
        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.conv1, 
                                 self.relu, 
                                 self.conv2, 
                                 self.relu, 
                                 self.conv3,
                                 self.relu, 
                                 self.conv4)
        
    def forward(self, x):

        output = self.net(x)

        return output


class DilatedConv(nn.Module):

    def __init__(self, 
                 num_inputs=64, 
                 dilation=1,
                 dropout=0.2):
                # h1=32, 
                # h2=16):
        
        """
        :param num_inputs: channels of residual
        :param dilations: 1, 2, 4, 8 ....
        :param h1: hidden units or channels
        :param h2: hidden units or channels

        :return: shape (batch_size, h2, xx)
        """
    
        super(DilatedConv, self).__init__()
        #self.hidden_units = [h1, h2, h2]

        #o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        self.conv1 = nn.Conv1d(in_channels=num_inputs, 
                               out_channels=num_inputs, 
                               kernel_size=3,
                               padding=0, 
                               dilation=dilation)

    def forward(self, x):

        output = self.conv1(x)

        return output


class ResBlock(nn.Module):

    def __init__(self, num_inputs=64, skip_channels=32, dilation=1):

        super(ResBlock, self).__init__()
        self.num_inputs = num_inputs
        self.skip_channels = skip_channels

        self.dilated = DilatedConv()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # residual
        self.conv1 = nn.Conv1d(in_channels=self.num_inputs, 
                               out_channels=self.num_inputs, 
                               kernel_size=1, dilation=1)
        
        # skip
        self.conv2 = nn.Conv1d(in_channels=self.num_inputs, 
                               out_channels=self.skip_channels, 
                               kernel_size=1, dilation=1)
        
    def forward(self, x):
            
        output = self.dilated(x)

        filters = self.tanh(output)
        gates = self.sigmoid(output)
        output = filters * gates
        
        residual = self.conv1(output)
        skip = self.conv2(output)

        x = residual + x[:, :, -residual.size(2):]
        
        return x, skip
        
        

class PostProcess(nn.Module):

    def __init__(self, num_inputs=32, h=16):
        
        """
        :param num_inputs: channels of skip
        :param h: hidden units
        
        :return: (batch_size, 1)
        """
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
                 n_steps_past=16, 
                 num_inputs=16,
                 dilation=[1,2,4,8]): #[32,32,32,64,64]):
            
        super(GluNet, self).__init__()

        self.num_inputs = num_inputs
        self.input_width = n_steps_past # n steps past
        
        self.causal = CausalConv()

        self.reslayers = nn.ModuleList([ResBlock(dilation=i) for i in dilation])
        
        self.dilated = DilatedConv()

        self.resblock = ResBlock()
        
        self.postprocess = PostProcess()


    def forward(self, x):
        
        residual = self.causal(x)
        skips = []

        for layer in self.reslayers:
            residual,skip = layer(residual)
            skips.append(skip)

        output = sum([s[:,:,-residual.size(2):] for s in skips])
        
        output = self.postprocess(output)

        return output

        




# lable transform and recover
        



