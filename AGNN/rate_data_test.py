#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time : 2020-08-30 16:15
# @Author : 呆子哥哥
# @File : rate_data_test.py
# @Software: PyCharm
# @Python Version: python 3.7
import dgl
import pandas as pd
import random
import numpy as np
import torch
import scipy.sparse as sp
path_base='../data/'
'''
注意：数据集中还有不少重复的
createGraph：创建图g；划分数据集；
'''

class DataBase:
    def __init__(self,dataset_name,rating_vals,ratio=0.8):
        '''
        数据集初始化
        :param dataset_name: string,like"ciao"
        :param ratio:训练集的比例
        '''
        # 数据集名字：如ciao
        self.dataset_name=dataset_name
        # 数据集比例[train,one_graph]
        self.ratio=ratio
        # 数据集长度
        self.data_length=0
        # 用户项目个数
        self.num_user=0
        self.num_item=0
        # 最高的评分
        self.max_score=0
        self.g=None
        self.possible_rating_values=rating_vals
        self.user_feature=None
        self.item_feature=None
        self.train_truth=None
        self.test_truth=None


    def divideData(self,dataset):
        '''
        划分数据集
        :param dataset: 交互数据集
        :return:
        '''
        # 打乱数据集
        np.random.shuffle(dataset)
        #训练集长度
        len_train=int(self.data_length*self.ratio)
        # 训练集数据
        train_data=dataset[0:len_train,0:3]
        # 测试集数据
        test_data=dataset[len_train: ,0:3]
        return train_data,test_data
    def createGraph(self):#ratio:训练集比例
        '''
        创建图B
        :return: 图
        '''
        # 读取信任数据
        # path=path_base+self.dataset_name+'\\interest.txt'
        # trust_data=pd.read_csv(path,header=None,names=['Src', 'Dst'])
        # trustor=torch.IntTensor(trust_data['Src'])-1
        # trustee=torch.IntTensor(trust_data['Dst'])-1
        # 读取交互数据
        path = path_base + self.dataset_name + '\\rate.txt'
        # interact_data = pd.read_csv(path, header=None, usecols=[0,1,2], dtype=int)
        interact_data = pd.read_csv(path, header=None, usecols=[0,1,2])
        dataset = interact_data.__array__()
        self.num_user, self.num_item, self.max_score = dataset.max(0).astype(int)
        dataset[:, 0] = dataset[:, 0] - 1
        dataset[:, 1] = dataset[:, 1] - 1

        # 交互总数据长度
        self.data_length=dataset.shape[0]
        # 划分数据集,训练集、测试集、验证
        train_data,test_data=self.divideData(dataset)
        self.train_truth=train_data[:,2]
        self.test_truth=test_data[:,2]


        # 读取gender信息
        path = path_base + self.dataset_name + '\\gender.npy'
        self.items_gender = torch.load(path)
        self.gender_len=self.items_gender.shape[1]
        self.item_feature=self.items_gender

        def f1(s):
            return (str(s)).replace('.','_')

        rating_row=train_data[:,0].astype(int)
        rating_col=train_data[:,1].astype(int)
        rating_values=train_data[:,2]
        data_dict=dict()
        for rating in self.possible_rating_values:
            ridx=np.where(rating_values==rating)
            rrow=rating_row[ridx]
            rcol=rating_col[ridx]
            data_dict.update({
                ('user',f1(str(rating)),'item'):(rrow,rcol),
                ('item',f1(str(rating)+'ed'),'user'):(rcol,rrow)
            })

        # 类型
        # 产生的异质图
        self.g = dgl.heterograph(data_dict, num_nodes_dict={'user': self.num_user, 'item': self.num_item})

        def _calc_norm(x):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.sqrt(x))
            return x.unsqueeze(1)

        user_ci = []
        user_cj = []
        movie_ci = []
        movie_cj = []
        for r in self.possible_rating_values:
            r=f1(r)
            user_ci.append(self.g[r+'ed'].in_degrees())
            movie_ci.append(self.g[r].in_degrees())
            user_cj.append(self.g[r].out_degrees())
            movie_cj.append(self.g[r+'ed'].out_degrees())
        user_ci = _calc_norm(sum(user_ci))
        movie_ci = _calc_norm(sum(movie_ci))
        user_cj = _calc_norm(sum(user_cj))
        movie_cj = _calc_norm(sum(movie_cj))
        self.g.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
        self.g.nodes['item'].data.update({'ci': movie_ci, 'cj': movie_cj})



        self.dec_g=dgl.heterograph({
            ('user', 'interact', 'item'): (torch.tensor(train_data[:, 0].astype(int)), torch.tensor(train_data[:, 1].astype(int))),  # 反过来
        }, num_nodes_dict={'user': self.num_user, 'item': self.num_item})
        # 测试集需要的图
        self.test_g = dgl.heterograph({
            ('user', 'interact', 'item'): (torch.tensor(test_data[:, 0].astype(int)), torch.tensor(test_data[:, 1].astype(int))),  # 反过来
        }, num_nodes_dict={'user': self.num_user, 'item': self.num_item})
        print("g ok")
    def setFeatures(self):
        pass
# data = DataBase('ml_100k')
# data.createGraph()