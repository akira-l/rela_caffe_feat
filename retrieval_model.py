from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pdb

class retrieval_model(nn.Module):
    def __init__(self, triple_size):
        super(retrieval_model, self).__init__()
        self.fc_outsize = triple_size
        '''
        self.avg_pool = nn.AvgPool3d(kernel_size=(3,3,3), 
                                     stride=(2,2,2), 
                                     padding=(0,0,0) )
        '''
        self.bn = nn.BatchNorm3d(num_features=300,
                                 track_running_stats=True)
        
        self.conv1_3d = nn.Conv3d(in_channels=300, 
                                  out_channels=100, 
                                  kernel_size=(1,1,1), 
                                  stride=(4,2,2), 
                                  padding=(0,0,0), 
                                  dilation=(1,1,1),  
                                  bias=True)

        self.conv2_3d = nn.Conv3d(in_channels=100, 
                                  out_channels=8, 
                                  kernel_size=(1,1,1), 
                                  stride=(2,2,2), 
                                  padding=(0,0,0), 
                                  dilation=(1,1,1),  
                                  bias=True)

        self.fc_insize = 2048 #5120#5120#1200*4#640
        self.retrieval_fc = nn.Linear(self.fc_insize, self.fc_outsize, bias=False)

    def trans_conv(self, in_layer, out_channel, kernel_size, strides):
        layer_shape = in_layer.shape()
        trans_conv = nn.ConvTranspose2d(in_channels=layer_shape[0], 
                                        out_channels=out_channel, 
                                        kernel_size=kernel_size, 
                                        stride=strides)
        return trans_conv

    def forward(self, inputs):
        bs = inputs.size(0)
        #l1 = self.avg_pool(inputs)
        inputs_bn = self.bn(inputs)
        l1 = F.relu(self.conv1_3d(inputs_bn))
        l2 = F.relu(self.conv2_3d(l1))
        #l1 = self.pool_3d(inputs)
        #l2 = F.relu(self.conv2_3d(l1))
        feat_to_fc = l2.view(bs, -1)
        retrieval_out = self.retrieval_fc(feat_to_fc)
        return l1, l2, retrieval_out


if __name__ == '__main__':
    tmp_feature = torch.rand(4, 300, 512, 7, 7)
    trans_mod = retrieval_model(128)
    tmp = trans_mod.test(tmp_feature)
    pdb.set_trace()
