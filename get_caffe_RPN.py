import os, sys
import torch 
import torch.nn as nn 
import torch.utils.data as data
import torch.autograd import Variable
import torch.nn.functional as F
import caffe 
import cv2

from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg, cfg_from_file

class ListDataset(data.Dataset):
    def __init__(self, roidb, roidb_reserve, roidb_rela_gt, sample_prob):
        self.roidb = roidb
        self.roidb_reserve = roidb_reserve
        self.roidb_rela_gt = roidb_rela_gt
        self.sample_prob = sample_prob
        self.use_roidb = []
        self.use_rela_gt = []
        self.use_rela_select_prob = []
        GPU_ID = 2
        caffe.set_device(GPU_ID)
        caffe.set_mode_gpu()
        net = None
        weights = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
        prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
        self.net = caffe.Net(prototxt, caffe.TEST, weights=weights)

        for roidb_counter in range(len(roidb_reserve)):
            if roidb_reserve[roidb_counter] == 1:
                self.use_roidb.append(roidb[roidb_counter])
                self.use_rela_gt.append(roidb_rela_gt[roidb_counter])
                self.use_rela_select_prob((roidb_rela_gt[roidb_counter]*sample_prob[:-1]).max().item())
        
    def __len__(self):
        return len(self.use_rela_select_prob)

    def __getitem__(self, idx):
        fname = self.use_roidb[idx]['image']
        im = cv2.imread(fname)
        scores, boxes, rel_scores = im_detect(self.net, im)
        rpn_feat = net.blobs['rpn/output'].data.copy()
        rela_gt = self.use_rela_gt[idx]
        usage_prob = self.use_rela_select_prob[idx]
        return rpn_feat, rela_gt, usage_prob

    def collect_fn(self, batch):
        feat = [x[0].unsqueeze(0) for x in batch]
        feat = torch.cat(feat, 0)
        rela_gt = [x[1].unsqueeze(0) for x in batch]
        rela_gt = torch.cat(rela_gt, 0)
        useage_prob = torch.tensor([x[2] for x in batch])
        rela_prob_sampler = torch.distributions.bernoulli.Bernoulli(useage_prob)
        rela_mask = rela_prob_sampler.sample().nonzero()
        while_counter = 0
        while rela_mask.dim() == 1:
            rela_mask = rela_prob_sampler.sample().nonzero()
            while_counter += 1
            if while_counter > 5:
                rela_mask = torch.ones_like(rela_mask)
        rela_gt_out = torch.zeros_like(rela_gt)
        rela_gt_out[:rela_mask.size(0), :, :] = rela_gt[rela_mask.nonzero().squeeze(1)]
        feat_out = torch.zeros_like(feat)
        feat_out[:rela_mask.size(0)] = feat[rela_mask.nonzero().squeeze(1)]
        return feat_out, rela_gt_out

    





'''



GPU_ID = 2
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()
net = None
weights = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

net = caffe.Net(prototxt, caffe.TEST, weights=weights)

im_file = roidb[]

im = cv2.imread(img_file)
scores, boxes, rel_scores = im_detect(net, im)
rpn_feat = net.blobs['rpn/output'].data.copy()

rpn/output   1, 512, 38, 50
pool5        137, 2048, 1, 1
pool5_flat   137, 2048

rpn_in_feat = Variable(torch.tensor(rpn_feat))

'''
