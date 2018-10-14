from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import datetime
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable 
import torch.nn.functional as F
import torchvision.utils as tushow
torch.backends.cudnn.enabled=False

from retrieval_model import retrieval_model
from retrieval_loss import retrieval_loss
from get_caffe_RPN import ListDataset
from data_table import data_table
#from visualize_utils import visualize
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='agent main')
    parser.add_argument('--epoch', dest='epoch',
                        default=5, type=int)
    parser.add_argument('--save_prefix', dest='save_prefix',
                        default="./retrieval_save", type=str)
    parser.add_argument('--bs', dest='batch_size',
                        default=16, type=int)
    
    parser.add_argument('--lr', type=float,
                        default=0.0007)
    parser.add_argument('--p_c', dest='precision_checkpoint',
                        default=10, type=int)
    # train val smalltrain smallval minitrain minival
    parser.add_argument('--data_version', dest='data_version',
                        default='minitrain', type=str)
    parser.add_argument('--optim', dest='optim',
                        default='adam', type=str)
    parser.add_argument('--single_gpu', dest='single_gpu', 
                        action='store_true')

    parser.add_argument('--epoch', dest='epoch',
                        default=10, type=int)
    args = parser.parse_args()
    return args

def train(args):
    # maintenance log
    save_suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    save_dir = args.save_prefix + save_suffix
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_title = 'log_'+save_suffix+'.txt'
    log_path = os.path.join(save_dir, log_title)
    precision_title = 'precision_'+save_suffix+'.txt'
    precision_path = os.path.join(save_dir, precision_title)
    train_info_log = 'train_info'+save_suffix+'.txt'
    train_info_path = os.path.join(save_dir, train_info_log)
    train_message_record(train_info_path, train_info_path)
    train_message_record(train_info_path, str(args))

    # prepare data
    data = data_table(args.data_version)
    data_package, process_package = data.get_table()
    # roidb reserve same length to roidb
    # roidb_rela_gt same length to roidb, gt tensor inside 
    #     same length to freq 
    # freq reserve relation pair sample probability

    imdb, roidb, ratio_list, ratio_index = data_package
    roidb_reserve, roidb_rela_gt, sample_prob = process_package
    
    rpn_list = ListDataset(roidb, roidb_reserve, roidb_rela_gt, sample_prob)
    rpn_loader = torch.utils.data.DataLoader(rpn_list, 
                                            batch_size=args.bs, 
                                            shuffle=True,
                                            num_workers=0,
                                            collect_fn=rpn_list.collect_fn)

    for epoch in range(args.epoch):
        for batch_idx, feat, gt in enumerate(rpn_loader):
            pdb.set_trace()



    
    triplet_size = len(process_package[2])
    # init training 
    get_loss = retrieval_loss(triplet_size)
    get_model = retrieval_model(triplet_size)
    rpn = get_RPN(data_package, process_package, batch_size=args.batch_size)
    data_size = rpn.get_len()
    train_message_record(train_info_path, str(get_model))
    

    if not args.single_gpu:
        get_model = torch.nn.DataParallel(get_model, device_ids=range(torch.cuda.device_count()))
        #rpn = torch.nn.DataParallel(rpn, device_ids=range(torch.cuda.device_count()))
    get_model.cuda()
    get_model.train()
    #rpn.cuda()

    if args.optim == 'adam':
        #optimizer = torch.optim.Adam(get_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, get_model.parameters()), lr=args.lr)#0.0002)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(get_model.parameters(), 
                        lr=args.lr, 
                        momentum=0.9, 
                        weight_decay=1e-4)
    else:
        raise NotImplementedError

    # training
    iteration = data_size//args.batch_size
    gt_record = []
    precision_checkpoint = args.precision_checkpoint
    for training_counter in range(iteration*args.epoch):
        set_times = training_counter-iteration*(training_counter//iteration)
        with torch.no_grad():
            '''
            if not args.single_gpu:
                fetch_data = rpn.module.get_data(set_times)
            else:
                fetch_data = rpn.get_data(set_times)
            '''
            fetch_data = rpn.get_data(set_times)
        if isinstance(fetch_data, int):
            continue
        feature, bbox_, relation_, width_, height_, usage, gt_ = fetch_data
        relation = Variable(relation_)
        height = Variable(height_)
        width = Variable(width_)
        bbox = Variable(bbox_)
        gt = Variable(gt_)

        if args.batch_size == 1:
            input_rpn = Variable(feature[1].unsqueeze(0))
        else:
            input_rpn = Variable(feature[1])
        _, _, retrieval_out = get_model(input_rpn)
        torch.cuda.synchronize()
        loss, re_gt = get_loss(retrieval_out, gt)
        gt_record.append(re_gt)

        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        print('training epoch: [%d / %d], iteration: [%d / %d]' % (training_counter//iteration, 
                args.epoch, 
                set_times, 
                iteration))
        print('loss %.4f' % (loss))

        write_log(log_path,
                  [training_counter//iteration,
                   args.epoch, 
                   set_times, 
                   iteration, 
                   round(float(loss), 4)],  
                   ['current_epoch',
                    'total_epoch', 
                    'current_times',
                    'iteration',
                    'loss'])
        
        if (set_times % precision_checkpoint == 0) and (set_times != 0):
            precision_, recall_, f_val_, thresh_ = count_p_r(retrieval_out, re_gt, step=0.1)
            write_log(precision_path,
                  [training_counter//iteration,
                   args.epoch, 
                   set_times, 
                   iteration, 
                   round(precision_[0], 4), 
                   round(recall_[0], 4), 
                   round(f_val_[0], 4),
                   round(thresh_[0], 4)],  
                   ['current_epoch',
                    'total_epoch', 
                    'current_times',
                    'iteration',
                    'precision',
                    'recall', 
                    'f_val', 
                    'thresh'])
        if (training_counter % (iteration) == 0 ) and (training_counter != 0):           
            torch.save(get_model.state_dict(), save_dir+'/retrieval_save'+str(training_counter//iteration)+'.pkl')

    gt_gather = torch.cat(gt_record, 0)
    count = []
    for gt_tmp in gt_gather:
        count.append((((gt_tmp == gt_gather).sum(1) == gt_tmp.size(0)).sum()-1).item())
        print('counting')
    coor_x = [x for x in range(len(count))]
    plt.plot(coor_x, count, 'r')
    plt.savefig(os.path.join(save_dir, 'count.png'))
    return save_dir, log_path, precision_path

######################################################
######################################################
######################################################
######################################################
def test(args, save_path):
    file_list = os.listdir(save_path)
    save_num = 1
    for name in file_list:
        if name[-4:] == '.pkl':
            if name.split('_')[0] == 'retrieval':
                tmp_num = int(re.findall('.*save(.*).pkl', name)[0])
                save_num = tmp_num if tmp_num > save_num else save_num
    retrieval_pkl = 'retrieval_save'+ str(save_num) +'.pkl'

    # prepare data
    data = data_table(args.data_version)
    data_package, process_package = data.get_table()
    # roidb reserve same length to roidb
    # roidb_rela_gt same length to roidb, gt tensor inside 
    #     same length to freq 
    # freq reserve relation pair sample probability

    triplet_size = len(process_package[2])
    # init training 
    rpn = get_RPN(data_package, process_package, batch_size=args.batch_size)

    get_loss = retrieval_loss(triplet_size).cuda()
    get_model = retrieval_model(triplet_size)
    get_model = torch.nn.DataParallel(get_model, device_ids=range(torch.cuda.device_count()))
    get_model.cuda()
    get_model.eval()
    get_model.load_state_dict(torch.load(os.path.join(save_path, retrieval_pkl)))

    '''
    pdb.set_trace() 
    vis_model = retrieval_model(triplet_size)
    vis1, vis2, vis3, vis4 = vis_model.feature()
    vis2_ = tushow.make_grid(vis2)
    save_fig_title = 'vis2.png'
    save_name = os.path.join(save_path, save_fig_title)
    tushow.save_image(vis2_, save_name)
    '''

    data_size = rpn.get_len()
    iteration = data_size//args.batch_size
    for test_counter in range(iteration):
        fetch_data = rpn.get_data(set_times=test_counter, raw_img=True)
        if isinstance(fetch_data, int):
            continue
        
        feature, bbox_, relation_, width_, height_, img_name, usage, gt_ = fetch_data
        relation = Variable(relation_)
        height = Variable(height_)
        width = Variable(width_)
        bbox = Variable(bbox_)
        gt = Variable(gt_)

        input_rpn = Variable(feature[1])
        # visualize v2
        vis_v1, vis_v2, retrieval_out = get_model(input_rpn)
        visualize(vis_v1, img_name, save_path)
        loss, re_gt = get_loss(retrieval_out, gt)
        
        print('test times: [%d / %d]' % (test_counter, iteration))
        print('loss %.4f' % (loss))

####################################################
####################################################
####################################################
####################################################

def count_p_r(cur_pred, gt, step, average=True):
    bs = cur_pred.size(0)
    pred_softmax = F.softmax(cur_pred, 1)
    pred_min_matrix = pred_softmax.min(1)[0].unsqueeze(1).repeat(1, cur_pred.size(1))
    pred_max_matrix = pred_softmax.max(1)[0].unsqueeze(1).repeat(1, cur_pred.size(1))
    pred = (pred_softmax-pred_min_matrix) / (pred_max_matrix-pred_min_matrix)
    thresh_list = [step*x for x in range(10)]
    precision_record = []
    recall_record = []
    thresh_record = []
    f_record = []
    for batch_counter in range(bs):
        gt_tmp = gt[batch_counter]
        pred_tmp = pred[batch_counter]
        precision_list = [((pred_tmp > thresh).float()*gt_tmp.float()).sum().float() / ((pred_tmp>thresh).float().sum()+1e-5) for thresh in thresh_list]
        recall_list = [((pred_tmp > thresh).float()*gt_tmp.float()).sum().float() / (gt_tmp.float().sum()+1e-5) for thresh in thresh_list]
        precision_list = [0 if x>1 else x for x in precision_list]
        recall_list = [0 if x>1 else x for x in precision_list]
        f_list = [p*r*2/(p+r+1e-5) for p,r in zip(precision_list, recall_list)]
        f_list = [0 if x>1 else x for x in f_list]
        ind = f_list.index(max(f_list))
        precision_record.append(precision_list[ind])
        recall_record.append(recall_list[ind])
        thresh_record.append(thresh_list[ind])
        f_record.append(f_list[ind])
    if average:
        return [(sum(precision_record) / len(precision_record)).item()],\
               [(sum(recall_record) / len(recall_record)).item()],\
               [(sum(f_record) / len(f_record)).item()], \
               [(sum(thresh_record) / len(thresh_record))]
    else:
        return precision_record, \
                recall_record, \
                f_record, \
                thresh_record

def train_message_record(log_path, content):
    file_obj = open(log_path, 'a')
    file_obj.write('args:'+content+'\n')
    file_obj.close() 

def write_log(log_path, record_list, name_list):
    file_obj = open(log_path, 'a')
    content = ''
    for val, name in zip(record_list, name_list):
        content += '_' + name + '=' + str(val) + '_'
    content += '\n'
    file_obj.write(content)
    file_obj.close()

def result_plot(target_path, save_path, key_word):
    file_obj = open(target_path, 'r')
    logs = file_obj.readlines()

    color_list = ['b', 'g', 'r', 'm', 'y', 'k']
    plt.cla()
    key_word_num = len(key_word)
    for key_word_counter in range(key_word_num):
        pattern = '.*'+key_word[key_word_counter]+'=(.*)'
        if key_word_counter < key_word_num -1:
            pattern += '__'
            pattern += key_word[key_word_counter+1][0]+'.*'
        else:
            pattern += '_.*'
        plot_var = [float(re.findall(pattern, x)[0]) for x in logs]
        coordx = [x for x in range(len(plot_var))]
        plt.plot(coordx, plot_var, color_list[key_word_counter])
    label = key_word
    plt.legend(label, loc='upper left')
    name = '_'.join(key_word) + '.png'
    plt.savefig(os.path.join(save_path, name))



if __name__ == '__main__':
    args = parse_args()
    save_path, log_path, precision_path = train(args)
    #save_path = './retrieval_save2018-09-29-06_51_57'
    #log_path = os.path.join(save_path, 'log_2018-09-29-06_51_57.txt')
    #precision_path = os.path.join(save_path, 'precision_2018-09-29-06_51_57.txt')
    result_plot(log_path, save_path, ['loss'])
    result_plot(precision_path, save_path, ['precision', 'recall', 'f_val', 'thresh'])
    test(args, save_path)

