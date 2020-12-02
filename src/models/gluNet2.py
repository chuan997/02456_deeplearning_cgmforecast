import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor

class Chomp1d(nn.Module):
    def __init__(self, length):

        """
        :param length: remove the last few units with a length

        :return: (batch_size, channels, l-length)
        """

        super(Chomp1d, self).__init__()
        self.length = length

    def forward(self, x):
        return x[:, :, :-self.length].contiguous()

class CausalConv(nn.Module):
    
    def __init__(self, 
                 num_inputs=4, 
                 causal_channels1=32, 
                 causal_channels2=64):
        
        """
        :param num_input: num of initial input channels (4)
        :param dilations: 1, 2, 4, 8 ....
        :param h1: hidden units or channels
        :param h2: hidden units or channels

        :return: shape (batch_size, h2, num_step_past)
        """
    
        super(CausalConv, self).__init__()
        self.hidden_units = [causal_channels1, causal_channels1, causal_channels2, causal_channels2]

        #padding=（kernel_size-1）*dilation
        self.conv1 = nn.Conv1d(in_channels=num_inputs, 
                               out_channels=self.hidden_units[0], 
                               kernel_size=2, padding=1, dilation=1)
        
        self.conv2 = nn.Conv1d(in_channels=self.hidden_units[0], 
                               out_channels=self.hidden_units[1], 
                               kernel_size=2, padding=1, dilation=1)
        
        self.conv3 = nn.Conv1d(in_channels=self.hidden_units[1], 
                               out_channels=self.hidden_units[2], 
                               kernel_size=2, padding=1, dilation=1)
        
        self.conv4 = nn.Conv1d(in_channels=self.hidden_units[2], 
                               out_channels=self.hidden_units[3], 
                               kernel_size=2, padding=1, dilation=1)
        
        self.relu = nn.ReLU()
        # chomp1d padding
        self.chomp = Chomp1d(length=1)

        self.net = nn.Sequential(self.conv1, self.relu, self.chomp, 
                                 self.conv2, self.relu, self.chomp, 
                                 self.conv3, self.relu, self.chomp, 
                                 self.conv4, self.chomp)
        
    def forward(self, x):

        output = self.net(x)

        return output


class DilatedConv(nn.Module):

    def __init__(self, 
                 num_inputs=64, 
                 dilation=1):
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
                               kernel_size=2,
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

        self.dilated = DilatedConv(num_inputs=num_inputs, dilation=dilation)
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

    def __init__(self, num_inputs=32, post_channels=16):
        
        """
        :param num_inputs: channels of skip
        :param h: hidden units
        
        :return: (batch_size, 1)
        """
        super(PostProcess, self).__init__()

        self.num_inputs = num_inputs
        self.hidden_units = post_channels

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
                 num_inputs=4,
                 dilation=[1,2,4,8], 
                 causal_channels1 = 32,
                 causal_channels2 = 64,
                 skip_channels = 32,
                 post_channels = 16): #[32,32,32,64,64]):
            
        super(GluNet, self).__init__()

        self.num_inputs = num_inputs
        self.input_width = n_steps_past # n steps past
        
        self.causal = CausalConv(causal_channels1=causal_channels1, causal_channels2=causal_channels2)

        self.reslayers = nn.ModuleList([ResBlock(num_inputs=causal_channels2, skip_channels=skip_channels, dilation=i) for i in dilation])
        
        #self.dilated = DilatedConv(num_inputs=causal_channels2)

        #self.resblock = ResBlock(num_inputs=causal_channels2, skip_channels=skip_channels)
        
        self.postprocess = PostProcess(num_inputs=skip_channels, post_channels=post_channels)


    def forward(self, x):
        
        residual = self.causal(x)
        #print(residual.shape)
        skips = []

        for layer in self.reslayers:
            residual,skip = layer(residual)
            skips.append(skip)
        #    print(layer)
        #    print('residual', residual.shape)
        #    print('skip', residual.shape)

        output = sum([s[:,:,-residual.size(2):] for s in skips])
        #print(output.shape)
        
        output = self.postprocess(output)

        return output

        




# lable transform and recover
        



