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

class retrieval_loss(nn.Module):
    def __init__(self, triplet_size):
        super(retrieval_loss, self).__init__()
        self.BCE_loss = nn.BCELoss()
        self.KLD_loss = nn.KLDivLoss(reduce=False)
        self.triplet_size = triplet_size

    def forward(self, pred, gt4loss):
        gt4loss = Variable(gt4loss.cuda())
        kld_tmp = self.KLD_loss(F.log_softmax(pred, 1), (((1/(gt4loss.sum(1)+1e-10))*gt4loss.t()).t()).cuda())
        kld_loss = kld_tmp[gt4loss.sum(1).nonzero().squeeze(1)]
        kld_loss = kld_loss.sum()/( gt4loss.sum(1).nonzero().nelement() )
        #return self.BCE_loss(F.sigmoid(pred), gt4loss.cuda())
        return kld_loss, (gt4loss>0).float()
