from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from PIL import Image
import os, sys
sys.path.append('./lib')
from scipy.misc import imread
import random
import time
from tqdm import tqdm
import pickle

import torch
import torch.utils.data as data

#from model.utils.config import cfg
#from model.utils.blob import prep_im_for_blob, im_list_to_blob
#from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from relation_roidb import combined_roidb

import pdb

class data_table(object):
    def __init__(self, data_version, reserve_thresh=5):
        super(data_table).__init__()
        self.data_version = data_version
        self.imdb_name = "vg_1600-400-500_"+data_version+"_xml1600"
        self.reserve_thresh = reserve_thresh
        self.imdb, self.roidb, self.ratio_list, self.ratio_index = combined_roidb(self.imdb_name)
        self.relation_list = self.get_relation_triplet(self.roidb)


    def get_table(self):
        # read a roidb relation first
        # i need to find unique triplet in this roidb first
        # rela_gather -> tensor others list
        all_rela_gather, all_rela_freq, roidb_rela_number_list = self.relation_table(self.relation_list)
        #allrelation_reserve_flag = self.select_relation(allrelation_freq, freq_threshold)
        roidb_reserve, roidb_rela_gt, sample_prob = self.generate_relation_gt(roidb_rela_number_list, all_rela_freq)
        return [self.imdb, self.roidb, self.ratio_list, self.ratio_index], [roidb_reserve, roidb_rela_gt, sample_prob]

    def allocate_probability(self, freq, gt_list):
        gt_len = len(gt_list)
        prob = torch.zeros(gt_len)
        prob_out = torch.zeros(gt_len+1)
        sum_gt = 0
        for pos_counter in range(gt_len):
            prob[pos_counter] = freq[gt_list[pos_counter]]
            sum_gt += freq[gt_list[pos_counter]]
        prob = 1-((prob-prob.min()+1) / prob.max())
        prob_out[:-1] = prob
        return prob_out**2

    def relation_table(self, relation_list):
        # fist read one to establish gather tensor
        # number 1, freq.append(1), roidb_number append 1
        roidb_num = len(relation_list)
        freq = []
        roidb_relation_number_list = []
        print("process every roidb ... ")
        count_bar = tqdm(total=roidb_num)
        if self.data_version == 'train':
            '''read pkl'''
            if os.path.exists('./largest_relation_table/roidb_relation_number_list.pkl'):
                pkl_file_list = open('./largest_relation_table/roidb_relation_number_list.pkl', 'rb')
                roidb_relation_number_list = pickle.load(pkl_file_list)
                pkl_freq = open('./largest_relation_table/preq.pkl', 'rb')
                freq = pickle.load(pkl_freq)
                relation_gather = torch.load('./largest_relation_table/relation_gather.pt')
                return relation_gather, freq, roidb_relation_number_list

        for roidb_counter in range(roidb_num):
            group_number_list = []
            relation_group = relation_list[roidb_counter]
            triplet_num = relation_group.size(0)
            for triplet_counter in range(triplet_num):
                triplet = relation_group[triplet_counter].unsqueeze(0)
                # init gather tensor
                if (roidb_counter == 0) and (triplet_counter == 0):
                    relation_gather = triplet
                    freq.append(1)
                    group_number_list.append(0)
                    continue
                check = ((triplet == relation_gather).sum(1).float() == 3).float()
                if check.sum().item() == 0:
                    # triplet not in gather tensor
                    relation_gather = torch.cat([relation_gather, triplet], 0)
                    freq.append(1)
                    group_number_list.append(relation_gather.size(0)-1)
                else:
                    # triplet already in gather tensor
                    #     find position first
                    pos = check.nonzero().squeeze(1).int().item()
                    freq[pos] += 1
                    group_number_list.append(pos)
            roidb_relation_number_list.append(group_number_list)
            count_bar.update(1)
        count_bar.close()
        if self.data_version == 'train':
            os.mkdir('./largest_relation_table')
            pkl_file_list = open('./largest_relation_table/roidb_relation_number_list.pkl', 'wb')
            pickle.dump(roidb_relation_number_list, pkl_file_list)
            pkl_freq = open('./largest_relation_table/preq.pkl', 'wb')
            pickle.dump(freq, pkl_freq)
            torch.save(relation_gather, './largest_relation_table/relation_gather.pt')
        return relation_gather, freq, roidb_relation_number_list

    def generate_relation_gt(self, roidb_number_list, all_rela_freq):
        reserve_flag = [1 if x>self.reserve_thresh else 0 for x in all_rela_freq]
        tmp_num_list = [x for x in range(len(reserve_flag))]
        gt_list = (torch.tensor(tmp_num_list)*torch.tensor(reserve_flag)).nonzero().squeeze(1).tolist()
        roidb_num = len(roidb_number_list)
        gt_len = len(gt_list)
        # +1 for no notable relation in img
        relation_gt = []
        roidb_reserve = []
        print("generate ground truth ... ")
        if self.data_version == 'train':
            if os.path.exists('./largest_relation_table/roidb_reserve.pkl'):
                roidb_pkl = open('./largest_relation_table/roidb_reserve.pkl', 'rb')
                roidb_reserve = pickle.load(roidb_pkl)
                relation_pkl = open('./largest_relation_table/relation_gt.pkl', 'rb')
                relation_gt = pickle.load(relation_pkl)
                return roidb_reserve, relation_gt, self.allocate_probability(all_rela_freq, gt_list)
        count_bar = tqdm(total=roidb_num)
        for roidb_counter in range(roidb_num):
            number_list = roidb_number_list[roidb_counter]
            roidb_reserve_flag = 0
            gt = torch.zeros(gt_len)
            for rela_item in number_list:
                if rela_item in gt_list:
                    roidb_reserve_flag = 1
                    gt[gt_list.index(rela_item)] += 1
            roidb_reserve.append(roidb_reserve_flag)
            relation_gt.append(gt)
            count_bar.update(1)
        count_bar.close()
        if self.data_version == 'train':
            reserve_pkl = open('./largest_relation_table/roidb_reserve.pkl', 'wb')
            pickle.dump(roidb_reserve, reserve_pkl)
            relation_pkl = open('./largest_relation_table/relation_gt.pkl', 'wb')
            pickle.dump(relation_gt, relation_pkl)
        return roidb_reserve, relation_gt, self.allocate_probability(all_rela_freq, gt_list)


    def get_relation_triplet(self, roidb):
        rel_list = []
        #pdb.set_trace()
        for roidb_counter in range(len(roidb)):
            if isinstance(roidb[roidb_counter], int):
                continue
            tmp_relation = roidb[roidb_counter]['gt_relations']
            tmp_cls = roidb[roidb_counter]['gt_classes']
            rel_shape = tmp_relation.shape
            rel_data = torch.zeros(rel_shape)
            rel_data[:,0] = torch.tensor(tmp_cls[tmp_relation[:,0]])
            rel_data[:,2] = torch.tensor(tmp_cls[tmp_relation[:,2]])
            rel_data[:,1] = torch.tensor(tmp_relation[:,1])
            rel_list.append(rel_data)
        # number of relation in rel_list fit the true number , without padding now
        return rel_list