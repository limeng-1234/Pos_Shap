#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/12/7 15:14
# @Author  : Limeng
# @FileName: FullyNet_Shapleyflow.py
# @Software: PyCharm
import os

import numpy as np
import torch
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# %%
class FullyNet_Shapleyflow(nn.Module):  # 继承父类，能够调用父类中的方法和属性等
    """ 计算每层神经元的shapley flow，目前只支持全连接网络。对于RNN，LSTM等复杂网络尚不支持.
    仅支持pytorch模型
    """

    def __init__(self, model, x):
        """ Build an explainers.
        Parameters
        ----------
        model : function 仅支持pytorch模型
        x     : background sample
        """
        super(FullyNet_Shapleyflow, self).__init__()
        self.model = model
        self.x_back = x.detach()

        self.init_hook()  # 每次重置 hook存储列表
        net_chilren = self.model.children()
        for child in net_chilren:
            if not isinstance(child, nn.ReLU6):
                child.register_forward_hook(hook=self.hook)
        self.model(self.x_back)  # 前向传播一次，会计算一次 hook：此处hook作用是计算非线性层输入和输出之间近似梯度
        self.module_base_hook = {'module_name': self.module_name, 'features_in_hook': self.features_in_hook,
                                 'features_out_hook': self.features_out_hook}

        ## self.init_hook()  # 每次重置 hook存储列表

    def init_hook(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def hook(self, module, fea_in, fea_out):
        # print("hooker working")
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None

    def calculate_shapley_flow(self, x_sample):
        list_shapley_flow = []
        for i in range(x_sample.shape[0]):
            temp = self.calculate_shapley_single(x_sample[i:i+1])
            list_shapley_flow.append(temp)
        return list_shapley_flow

    def calculate_shapley_single(self, x_sample):
        """ 计算被解释样本x_sample在每层神经元的shapley flow.
        返回一个list，包含每层神经元的shapley flow，list中每个元素以三维数组表示：（neuron(in), class(输出的类别)，neuron(out)），其中， neuron是相应的layer层
        Parameters
        ----------
        x_sample : 被解释样本
        """
        self.init_hook()  # 重置hook存储列表,否则该列表会继续之前的存储
        self.x_sample = x_sample.detach()
        self.model(self.x_sample)
        self.module_sample_hook = {'module_name': self.module_name, 'features_in_hook': self.features_in_hook,
                                   'features_out_hook': self.features_out_hook}
        # 计算每一层的权重   遍历所有名字
        list_weight = []
        for step, item in enumerate(self.module_sample_hook['module_name']):
            if item == torch.nn.modules.linear.Linear:
                weight = self.model[step].weight
                list_weight.append(weight)
            else:
                weight = torch.div(
                    self.module_sample_hook['features_out_hook'][step] - self.module_base_hook['features_out_hook'][
                        step],
                    self.module_sample_hook['features_in_hook'][step][0] - self.module_base_hook['features_in_hook'][
                        step][0]).t()
                weight = torch.diag_embed(weight.squeeze(dim=1))  # 非线性函数是1 by 1的，所以将其转换成对角矩阵
                list_weight.append(weight)
        temp = (self.x_sample-self.x_back)
        aa = temp.expand(list_weight[-1].data.shape[0], temp.shape[1])
        chain_weight = list_weight[0].data
        for i in range(len(list_weight)):
            if i > 0:
                chain_weight = torch.matmul(list_weight[i].data, chain_weight)
            # aa = list_weight[i].data * aa
        aa = chain_weight * aa
        aa[torch.isnan(aa)] = 0
        return aa



