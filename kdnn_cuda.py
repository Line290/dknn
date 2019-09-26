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
import time
# ALPHA = 1. - 1./60000
ALPHA = 0.99
# VECTOR_SIMILARITY_METRICS = 'cosine'
VECTOR_SIMILARITY_METRICS = 'Euclid'
THRESHOLD = 10

class QKNet(nn.Module):
    def __init__(self, input_sizes=(None, 1, 28, 28),
                 hidden_channel_dims = [32, 64, 7*7*64, 1024],
                 kernel_sizes = [5, 5],
                 hidden_out_dims = [14, 7],
                 nb_class=10,
                 nb_k_center=200,
                 device='cpu',
                 keep_center_id=False):
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
        self.keep_center_id = keep_center_id
        # center_dict: keep each layer's feature centers
        # table_dict: count number of feature vectors which belongs to a specific center
        self.center_dict, self.table_dict = self._init_center_and_table_dict()
        self.center_id_saver = []
        # Net
        self.conv1 = nn.Conv2d(self.nb_channel,
                               self.hidden_channel_dims[0],
                               self.kernel_sizes[0],
                               padding=2)
        self.conv2 = nn.Conv2d(self.hidden_channel_dims[0],
                               self.hidden_channel_dims[1],
                               self.kernel_sizes[1],
                               padding=2)
        # self.conv3 = nn.Conv2d(self.hidden_channel_dims[1],
        #                        self.hidden_channel_dims[2],
        #                        self.kernel_sizes[2],
        #                        padding=2)
        # self.conv4 = nn.Conv2d(self.hidden_channel_dims[2],
        #                        self.hidden_channel_dims[3],
        #                        self.kernel_sizes[3],
        #                        padding=2)
        self.fc1 = nn.Linear(self.hidden_channel_dims[-2], self.hidden_channel_dims[-1])
        self.fc2 = nn.Linear(self.hidden_channel_dims[-1], self.nb_class)
        self.max_pool2d = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        # self.prelu = nn.PReLU()
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.NN_Q = self.NN_quantization.apply

    def _init_center_and_table_dict(self):
        center = {}
        table = {}

        for layer, dim in enumerate(self.hidden_out_dims):
            center[layer] = torch.randn(self.hidden_channel_dims[layer],
                                            self.nb_k_center, dim*dim).to(self.device)
            table[layer] = torch.zeros((self.hidden_channel_dims[layer],
                                     self.nb_k_center, 1), dtype=torch.int64).to(self.device)
        return center, table

    class NN_quantization(Function):
        @staticmethod
        def forward(ctx, input, layer_idx, device, training, keep_center_id, center_dict, table_dict, center_id_saver):
            # feature map, shape (batch_size, nb_channel, height, width)
            # center, shape (nb_channel, nb_k_center, height*width)
            # table, shape (nb_channel, nb_k_center, 1)
            nb_channel, nb_k_center, feat_dim = center_dict[layer_idx].size()
            batch_size = input.size(0)
            input = input.view(batch_size, nb_channel, -1)

            # find KNN
            cos_dist = None
            if 'cosine' in VECTOR_SIMILARITY_METRICS:
                cos_dist = 1. - torch.sum(torch.unsqueeze(input, dim=2) * center_dict[layer_idx], dim=-1)
                # cos_dist = torch.einsum("abc,cd->abd", (input, center.transpose(0, 1)))
            elif 'Euclid' in VECTOR_SIMILARITY_METRICS:
                cos_dist = torch.sum((torch.unsqueeze(input, dim=2) - center_dict[layer_idx]) ** 2, dim=-1)
                
            if training is False:
                # set invalid center's cosine distance as 1
                # if 'cosine' in VECTOR_SIMILARITY_METRICS:
                #     cos_dist[:, table_dict[layer_idx].squeeze(-1) < THRESHOLD] = 1.
                # elif 'Euclid' in VECTOR_SIMILARITY_METRICS:
                #     cos_dist[:, table_dict[layer_idx].squeeze(-1) < THRESHOLD] = 1e10
                pass
            indices = torch.argmin(cos_dist, dim=-1, keepdim=False)  # shape: (b_s, c_s)

            # update center

            # save
            # if layer_idx == 1:
            #     center = center_dict[layer_idx].cpu().numpy()
            #     indices_cpu = indices.cpu().numpy()
            #     input_cpu = input.cpu().numpy()
            #     table = table_dict[layer_idx].cpu().numpy()
            #     new_center = np.zeros_like(center)
            #     print('start save')
            #     np.save('center.npy', center)
            #     np.save('indices_cpu.npy', indices_cpu)
            #     np.save('input_cpu.npy', input_cpu)
            #     np.save('table.npy', table)
            #     # np.save('table.npy', table)

            if training:
                #####################################
                ########   vectorization   ##########
                #####################################
                # if PyTorch version>=1.2
                # tmp_center = torch.unsqueeze(input, dim=2) * F.one_hot(indices, nb_k_center).unsqueeze(dim=-1)
                # else
                one_hot = torch.cuda.FloatTensor(batch_size*nb_channel, nb_k_center).fill_(0)
                one_hot = one_hot.scatter_(1, indices.view(-1, 1), 1)
                one_hot = one_hot.view(batch_size, nb_channel, nb_k_center, 1)
                tmp_center = torch.unsqueeze(input, dim=2) * one_hot

                tmp_center = tmp_center.permute(1, 2, 0, 3)
                tmp_sum_center = tmp_center.sum(dim=-2)
                count = ((tmp_center.sum(dim=-1) > 0.).int()).sum(dim=-1, keepdim=True)


                # moving average center
                # tmp_center = tmp_sum_center / (count.float() + 1e-12)
                # center_dict[layer_idx] = center_dict[layer_idx] * ALPHA + (1 - ALPHA) * tmp_center
                # center_dict[layer_idx] = F.normalize(center_dict[layer_idx], p=2, dim=-1)
                # table_dict[layer_idx] += count

                # average center
                # print(center_dict[layer_idx].size(), table_dict[layer_idx].unsqueeze(dim=-1).size(), tmp_sum_center.size())
                center_dict[layer_idx] = center_dict[layer_idx] * table_dict[layer_idx].float() + tmp_sum_center
                table_dict[layer_idx] += count
                center_dict[layer_idx] = center_dict[layer_idx] / (table_dict[layer_idx].float() + 1e-12)

                #####################################
                ########       naive       ##########
                #####################################
                # for i in range(nb_channel):
                #     for j in range(nb_k_center):
                #         tmp_feature_vectors = input[:, i, :][indices[:, i] == j].view(-1, feat_dim)
                #         tmp_feature_len = tmp_feature_vectors.size(0)
                #         if tmp_feature_len == 0:
                #             continue
                #         tmp_count = table_dict[layer_idx][i, j]
                #         table_dict[layer_idx][i, j] += tmp_feature_len
                #         if tmp_count == 0:
                #             tmp = tmp_feature_vectors.mean(dim=0)
                #         else:
                #             # moving average
                #             tmp = tmp_feature_vectors.mean(dim=0)
                #             tmp = center_dict[layer_idx][i, j] * ALPHA + (1 - ALPHA) * tmp
                #         center_dict[layer_idx][i, j] = F.normalize(tmp.view(1, -1), p=2)

                # if layer_idx==1:
                #     np.save('new_center.npy', new_center)
                #     np.save('new_new_center.npy', center_dict[layer_idx].cpu().numpy())
                #     np.save('new_table.npy', table_dict[layer_idx].cpu().numpy())
                #     print('save all')
                #     time.sleep(5)
            else:
                # print('testing')
                if keep_center_id:
                    center_id_saver.append(indices)
                else:
                    pass

            result = center_dict[layer_idx][range(nb_channel), indices]
            # ctx.save_for_backward(ctx, input)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None, None, None, None, None, None, None


    def set_center(self, center, table):
        '''
        :param center: list of torch Tensor, dtype: float32
        :param table: torch Tensor: int64

        Use torch.save(tensor, 'file.pt') and torch.load('file.pt')
        '''
        # self.center_dict = torch.from_numpy(center).to(self.device)
        # self.table_dict = torch.from_numpy(table).to(self.device)
        # self.center_dict = center.to(self.device)
        for layer, dim in enumerate(self.hidden_out_dims):
            self.center_dict[layer] = center[layer].to(self.device)
            self.table_dict[layer] = table[layer].to(self.device)

    def reset_center_id_saver(self):
        self.center_id_saver = []

    def knn_layer(self, x, layer_idx=0):
        batch_size, channel_size, h, w = x.size()
        x = F.normalize(x.view(batch_size, channel_size, -1),
                        p=2, dim=2)
        # quantize
        x = self.NN_Q(x, layer_idx,
                      self.device,
                      self.training,
                      self.keep_center_id,
                      self.center_dict,
                      self.table_dict,
                      self.center_id_saver)

        x = x.view(batch_size, channel_size, h, w)
        return x

    def forward(self, x):
        # global table_dict, center_dict
        x = self.relu(self.conv1(x))
        x = self.max_pool2d(x)
        x = self.knn_layer(x, layer_idx=0)

        x = self.relu(self.conv2(x))
        x = self.max_pool2d(x)
        x = self.knn_layer(x, layer_idx=1)

        # x = self.relu(self.conv3(x))
        # x = self.max_pool2d(x)
        # x = self.knn_layer(x, layer_idx=2)
        #
        # x = self.relu(self.conv4(x))
        # x = self.knn_layer(x, layer_idx=3)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # print(table_dict[0][:10])
        # print(center_dict[0][0][:10])
        # print(table_dict[0][0])
        return x
        
