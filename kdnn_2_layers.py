#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
'''

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch
import random
from scipy import spatial
ALPHA = 0.99
VECTOR_SIMILARITY_METRICS = 'cosine'
global center_dict, table_dict

class NN_quantization(Function):
    @staticmethod
    def forward(ctx, input, layer_idx, device, mode):
        global center_dict, table_dict
        # get KNN info.
        center = torch.from_numpy(center_dict[layer_idx]).to(device)
        table = torch.from_numpy(table_dict[layer_idx]).to(device)
        # detach so we can cast to NumPy
        # feature map, shape (batch_size, channel_size, height, width)
        # center, shape (nb_channel, nb_k_center, height*width)
        # table, shape (nb_channel, nb_k_center, 1)
        # input, center, table = \
        #     input.detach().numpy(), center.detach().numpy(), table.detach().numpy()
        # input = input.reshape(input.shape[0], input.shape[1], -1)
        input = input.view(input.size(0), input.size(1), -1)
        # print(input.size())
        # find KNN
        cos_dist = None
        if 'cosine' in VECTOR_SIMILARITY_METRICS:
            cos_dist = 1. - torch.sum(torch.unsqueeze(input, dim=2) * center, dim=-1)
            # cos_dist = torch.einsum("abc,cd->abd", (input, center.transpose(0, 1)))

        indices = torch.argmin(cos_dist, dim=-1, keepdim=False) # shape: (b_s, c_s, 1)

        # update center
        input_, center_, table_ = \
            input.cpu().numpy(), center.cpu().numpy(), table.cpu().numpy()
        indices = indices.cpu().numpy()
        if mode == 'train':
            for i in range(center_.shape[0]):
                for j in range(center_.shape[1]):
                    tmp_feature_vectors = input_[:, i, :][indices[:, i]==j].reshape(-1, center_.shape[2])
                    if tmp_feature_vectors.shape[0] == 0:
                        continue
                    tmp_count = table_[i, j]
                    table_[i, j] = table_[i, j] + tmp_feature_vectors.shape[0]
                    if tmp_count == 0:
                        tmp = tmp_feature_vectors.sum(axis=0) / table_[i, j]
                    else:
                        # tmp = (center_[i, j]*tmp_count + tmp_feature_vectors.sum(axis=0)) / table_[i, j]
                        # moving average
                        tmp = center_[i, j]*ALPHA + (1 - ALPHA)*tmp_feature_vectors.sum(axis=0)
                    center_[i, j] = tmp / np.linalg.norm(tmp, ord=2)
        elif mode == 'test':
            pass
        else:
            pass
        # print(center_.shape, indices.shape)
        result = center_[range(center_.shape[0]), indices]
        # print(result.shape)
        # update info.
        center_dict[layer_idx] = center_
        table_dict[layer_idx] = table_
        # ctx.save_for_backward(ctx, input)
        return torch.as_tensor(result, dtype=input.dtype).to(device)
               # torch.as_tensor(center_, dtype=center.dtype), \
               # torch.as_tensor(table_, dtype=table.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

class QKNet(nn.Module):
    def __init__(self, input_sizes=(None, 1, 28, 28),
                 hidden_channel_dims = [32, 64, 7*7*64, 1024],
                 kernel_sizes = [5, 5],
                 hidden_out_dims = [14, 7],
                 nb_class=10,
                 nb_k_center=10,
                 device='cpu',
                 mode='train'):
        super(QKNet, self).__init__()
        self.nb_channel = input_sizes[1]
        self.img_h = input_sizes[2]
        self.img_w = input_sizes[3]
        self.hidden_channel_dims = hidden_channel_dims
        self.kernel_sizes = kernel_sizes
        self.hidden_out_dims = hidden_out_dims
        self.nb_class = nb_class
        self.nb_k_center = nb_k_center
        self.device = device
        self.mode = mode
        # center_dict: keep each layer's feature centers
        # table_dict: count number of feature vectors which belongs to a specific center
        self.center_dict, self.table_dict = self._init_center_and_table_dict()
        global center_dict, table_dict
        center_dict = self.center_dict
        table_dict = self.table_dict
        # Net
        self.conv1 = nn.Conv2d(self.nb_channel,
                               self.hidden_channel_dims[0],
                               self.kernel_sizes[0],
                               padding=2)
        self.conv2 = nn.Conv2d(self.hidden_channel_dims[0],
                               self.hidden_channel_dims[1],
                               self.kernel_sizes[1],
                               padding=2)
        self.fc1 = nn.Linear(self.hidden_channel_dims[2], self.hidden_channel_dims[3])
        self.fc2 = nn.Linear(self.hidden_channel_dims[3], self.nb_class)
        self.max_pool2d = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.NN_Q = NN_quantization.apply
        
    def _init_center_and_table_dict(self,):
        center = {}
        table = {}
        # if self.mode == 'test':
        #     center = np.load('../center_dict.npy').item()
        #     table = np.load('../table_dict.npy').item()
        # elif self.mode == 'train':
        for layer, dim in enumerate(self.hidden_out_dims):
            tmp = (np.random.rand(self.hidden_channel_dims[layer],
                                            self.nb_k_center, dim*dim)*0.01).astype(np.float32)
            center[layer] = tmp / np.linalg.norm(tmp, ord=2, axis=-1, keepdims=True)
            table[layer] = np.zeros((self.hidden_channel_dims[layer],
                                     self.nb_k_center, 1), dtype=np.int64)
        return center, table

    def set_center(self, center, table):
        global center_dict, table_dict
        center_dict, table_dict = center, table

    def knn_layer(self, x, layer_idx=0):
        global center_dict, table_dict
        batch_size, channel_size, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        x = F.normalize(x.view(x.size(0), x.size(1), -1),
                        p=2, dim=2)
        # # get KNN info.
        # center = torch.from_numpy(self.center_dict[layer_idx]).to(self.device)
        # table = torch.from_numpy(self.table_dict[layer_idx]).to(self.device)
        # quantize
        x = self.NN_Q(x, layer_idx, self.device, self.mode)

        # # update center and table
        # self.center_dict[layer_idx] = center.detach().numpy()
        # self.table_dict[layer_idx] = table.detach().numpy()
        self.center_dict = center_dict
        self.table_dict = table_dict
        x = x.view(batch_size, channel_size, h, w)
        return x.to(self.device)

    def forward(self, x):
        # global table_dict, center_dict
        x = self.relu(self.conv1(x))
        x = self.max_pool2d(x)
        x = self.knn_layer(x, layer_idx=0)
        x = self.relu(self.conv2(x))
        x = self.max_pool2d(x)
        x = self.knn_layer(x, layer_idx=1)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # print(table_dict[0][:10])
        # print(center_dict[0][0][:10])
        # print(table_dict[0][0])
        return x
        
