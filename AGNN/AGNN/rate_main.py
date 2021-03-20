#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020-08-30 19:41
# @Author : 呆子哥哥
# @File : rate_main.py
# @Software: PyCharm
# @Python Version: python 3.7
# 调好的，可以用的the best MAE:0.7097 RMSE:0.9034

# epoch:236 | MAE:0.7188 | RMSE:0.9088
# epoch:237 | MAE:0.7237 | RMSE:0.9134
# the best MAE:0.7153 RMSE:0.9077
# embed+属性

import argparse
import itertools
import numpy as np
from dgl.nn import GraphConv
rating_vals=[]
from AGNN.rate_data_test import DataBase
from dgl.data import register_data_args
import torch.nn as nn
import torch
from AGNN.model import *
from AGNN.utils import  *
import time
import math
# device = "cpu"
import torch as th

# epoch:239 | MAE:0.7224 | RMSE:0.9102
# epoch:239 | MAE:0.7224 | RMSE:0.9102
# epoch:240 | MAE:0.7295 | RMSE:0.9173
# the best MAE:0.7139 RMSE:0.9036
class myModel(nn.Module):
    def __init__(self,dataset,args):
        super(myModel, self).__init__()
        self._act =args.model_activation
        self.user_embed = nn.Embedding(dataset.num_user, args.embed_units)
        self.item_embed = nn.Embedding(dataset.num_item, args.embed_units)
        self.num_user=dataset.num_user
        self.num_item=dataset.num_item
        data_dict={}
        for rating in rating_vals:
            data_dict[(str(rating)+'ed').replace('.','_')]=GraphConv(dataset.gender_len, dataset.gender_len)

        self.genre_conv=dgl.nn.HeteroGraphConv(
            data_dict#,allow_zero_in_degree=True
        )

        self.encoder = MyLayer(
            args.embed_units+dataset.gender_len,args.embed_units+dataset.gender_len,args.gcn_agg_units,args.gcn_out_units,
            rating_vals,args.gcn_dropout,args.gcn_agg_accum,self._act,self._act)
        self.pred=  MLPPredictor(args.gcn_out_units,args.gcn_dropout)

    def forward(self,g,dec_g ,ufeat=None,ifeat=None):
        ufeat = self.genre_conv(g, {'item': ifeat})['user']
        ufeat_em=self.user_embed(th.tensor([i for i in range(self.num_user)]))
        ifeat_em = self.item_embed(th.tensor([i for i in range(self.num_item)]))
        h_dict = self.encoder(g, {'user':torch.cat((ufeat_em,ufeat),1),'item':torch.cat((ifeat_em,ifeat),1)})
        pre=self.pred(dec_g,h_dict)
        return pre

def train(args):
    print(args)
    global rating_vals
    rating_vals = [1, 2, 3, 4, 5]
    if (args.data_name=='ml_latest'):
        rating_vals=[1,1.5,2,2.5,3,3.5,4,4.5,5]
    dataset = DataBase(args.data_name,rating_vals,ratio=args.data_train_ratio)
    dataset.createGraph()
    print("Loading data finished ...\n")

    best_MAE=1000
    best_RMSE=1000
    ### build the net
    net = myModel(dataset,args)
    # nd_possible_rating_values = th.FloatTensor(dataset.possible_rating_values).to(args.device)
    rating_loss_net = nn.MSELoss()
    learning_rate = args.train_lr
    # optimizer = torch.optim.Adam(net.parameters(),
    #                                  lr=learning_rate)
    #                                  # weight_decay=args.weight_decay)
    optimizer = get_optimizer(args.train_optimizer)(net.parameters(), lr=learning_rate)
    print("Loading network finished ...\n")

    ### perpare training data
    train_truth=torch.tensor(dataset.train_truth).float()
    print("Start training ...")
    dur = []
    for iter_idx in range(1, args.train_max_iter+1):

        net.train()
        pred_ratings = net(dataset.g, dataset.dec_g,
                           dataset.user_feature, dataset.item_feature)
        loss = rating_loss_net(pred_ratings.flatten(), train_truth).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
        optimizer.step()
        net.eval()
        with th.no_grad():
            pred_ratings = net(dataset.g, dataset.test_g,
                           dataset.user_feature, dataset.item_feature)
        real_pred_ratings = pred_ratings.flatten()
        minus=th.abs(real_pred_ratings - dataset.test_truth)
        MAE=th.mean(minus)
        RMSE=th.sqrt(th.mean(minus**2))
        # print(loss.item())
        print('epoch:{:d} |trainLoss:{:.4f} | test:MAE:{:.4f} | RMSE:{:.4f}'.format(iter_idx,loss.item(),MAE.item(),RMSE.item()))
        if RMSE.item() < best_RMSE:
            best_RMSE = RMSE.item()
            best_MAE = MAE.item()
            count = 0
        else:
            count = count + 1
        if count > 100:
            break
    print('the best MAE:{:.4f} RMSE:{:.4f}'.format(best_MAE, best_RMSE))
    print('embed+属性')






def config():
    parser = argparse.ArgumentParser(description='GCMC')

    parser.add_argument('--data_name', default='ml_100k', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml_latest')
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--gcn_dropout', type=float, default=0.5)
    parser.add_argument('--embed_units', type=int, default=512)
    parser.add_argument('--gcn_agg_units', type=int, default=128)
    parser.add_argument('--gcn_agg_accum', type=str, default="stack")
    parser.add_argument('--gcn_out_units', type=int, default=128)
    parser.add_argument('--train_optimizer', type=str, default="adam")#优化器
    parser.add_argument('--train_max_iter', type=int, default=1200)
    parser.add_argument('--data_train_ratio', type=float, default=0.8)
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', default=123, type=int)

    # parser.add_argument('--device', default='0', type=int,
    #                     help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    # parser.add_argument('--save_dir', type=str, help='The saving directory')
    # parser.add_argument('--save_id', type=int, help='The saving log id')
    # parser.add_argument('--silent', action='store_true')
    # parser.add_argument('--data_train_ratio', type=float, default=0.8)  ## for ml-100k the test ration is 0.2
    # parser.add_argument('--use_one_hot_fea', action='store_true', default=True)
    # parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    # parser.add_argument('--gen_r_num_basis_func', type=int, default=2)
    # parser.add_argument('--train_log_interval', type=int, default=1)
    # parser.add_argument('--train_valid_interval', type=int, default=1)
    # parser.add_argument('--train_grad_clip', type=float, default=1.0)
    # parser.add_argument('--train_min_lr', type=float, default=0.001)
    # parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    # parser.add_argument('--train_decay_patience', type=int, default=50)
    # parser.add_argument('--train_early_stopping_patience', type=int, default=100)
    # parser.add_argument('--share_param', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = config()
    print(args.train_lr)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    train(args)














# def main(args):
#     data = DataBase(args.dataset_name,ratio=0.8)
#     data.createGraph()
#     g = data.g
#     test_g=data.test_g
#     # print("""
#     # ----Data statisstics----'
#     # #num_user=%d
#     # #num_item=%d
#     # num_interact=%d
#     # #Train  samples %d
#     # #Test samples %d
#     # """ % (data.num_user, data.num_item,
#     #        # g.number_of_edges('interact'),
#     #        data.data_length*data.ratio,
#     #        data.data_length*(1-data.ratio))
#     #       )
#     args.src_in_units=943
#     args.dst_in_units=1682
#
#     model = myModel(args.src_in_units,args.dst_in_units,args.agg_units,args.out_units,args.dropout)
#     # 损失函数，二分类
#     loss_fcn = nn.CrossEntropyLoss()
#     # 优化器
#     optimizer = torch.optim.Adam(model.parameters(),
#                                  lr=args.lr,
#                                  weight_decay=args.weight_decay)
#
#
#     for epoch in range(args.n_epochs):
#         t0 = time.time()
#         model.train()
#         pred_score = model(g, data.dec_g,data.user_feature,data.item_feature)
#         loss = loss_fcn(pred_score,data.train_labels)
#         optimizer.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
#         optimizer.step()
#         if epoch == args.n_epochs - 2:
#             print()
#         print('Epoch:{:05d}    |   Time(s){:.4f}|    Loss{:.4f}|A'.format(epoch, time.time() - t0, loss.item()))
#
#         # 测试集
#         model.eval()
#         with th.no_grad():
#             pred_ratings = model(g, test_g, data.user_feature, data.item_feature)
#         real_pred_ratings = (th.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
#         minus=th.abs(real_pred_ratings - data.test_truth)
#         MAE=th.mean(minus)
#         RMSE=th.sqrt(th.mean(minus**2))
#         print(MAE,RMSE)
#
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='recommderGCN')
#     register_data_args(parser)
#
#     # # 数据集
#     parser.add_argument("--dataset_name", type=str, default='ml_100k',  # ciao  epinions
#                         help="the dataset's name")
#
#     # dropout设置
#     parser.add_argument("--dropout", type=float, default=0.7,
#                         help="dropout probability")
#
#     parser.add_argument('--agg_units',type=int,default=500)
#     parser.add_argument('--out_units', type=int, default=75)
#
#     # 学习率
#     parser.add_argument("--lr", type=float, default=1e-1,
#                         help="learning rate")
#     # 训练epoch
#     parser.add_argument("--n-epochs", type=int, default=300,
#                         help="number of training epochs")
#
#     # embeding层
#     parser.add_argument("--n-ebd", type=int, default=1024,#32
#                         help="number of hidden ebd units")
#     # 网络深度
#     parser.add_argument("--n-layers", type=int, default=1,
#                         help="number of hidden gcn layers")
#     # L2正则化，即权重衰减
#     parser.add_argument("--weight-decay", type=float, default=1e-1,  # 5e-4
#                         help="Weight for L2 loss")
#     # parser.add_argument('--self-loop', action='store_true', default=False, help='graph self-loop(default=False)')
#     parser.add_argument('--train_grad_clip', type=float, default=1.0)
#     args = parser.parse_args()
#
#     print(args)
#     main(args)












