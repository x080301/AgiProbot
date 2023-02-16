#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: bixuelei
@Contact: xueleibi@gmail.com
@File: model.py
@Time: 2022/1/15 17:11 PM
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from torch.autograd import Variable

from para_extracter.utilities.utilities import get_neighbors


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):  # input (B,3,N)
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # (B,3,N)->(B,64,N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B,64,N)->(B,128,N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B,128,N)->(B,1024,N)
        x, _ = torch.max(x, 2, keepdim=True)  # (B,1024,N)->(B,1024,1)
        x = x.view(-1, 1024)  # (B,1024,1)->(B,1024)

        x = F.relu(self.bn4(self.fc1(x)))  # (B,1024)->(B,512)
        x = F.relu(self.bn5(self.fc2(x)))  # (B,512)->(B,256)
        x = self.fc3(x)  # (B,256)->(B,9)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden  # (B,9)
        x = x.view(-1, 3, 3)  # (B,9)->(B,3,3)
        return x  # output (B,3,3)


class SA_Layer_Single_Head(nn.Module):
    def __init__(self, channels):
        super(SA_Layer_Single_Head, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # input (B,N,128)
        x = x.permute(0, 2, 1)  # (B,N,128)->(B,128,N)
        x_q = self.q_conv(x).permute(0, 2, 1)  # (B,128,N)->(B,32,N)->(B,N,128)
        x_k = self.k_conv(x)  # (B,128,N)->(B,32,N)
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # (B,N,128)@(B,128,N)->(B,N,N)
        attention = self.softmax(energy)  # (B,N,N)->(B,N,N)
        attention = attention / (1e-6 + attention.sum(dim=1, keepdims=True))  # (B,N,N)->(B,N,N)
        x_r = x @ attention  # (B,128,N)@(B,N,N)->(B,128,N)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))  # (B,128,N)->(B,128,N)
        x = x + x_r
        x = x.permute(0, 2, 1)  # (B,128,N)->(B,N,128)
        return x  # output (B,N,128)


class PCT_semseg(nn.Module):
    def __init__(self, args):
        super(PCT_semseg, self).__init__()
        self.args = args
        self.k = args.k
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.sa1 = SA_Layer_Single_Head(128)
        self.sa2 = SA_Layer_Single_Head(128)
        self.sa3 = SA_Layer_Single_Head(128)
        self.sa4 = SA_Layer_Single_Head(128)
        self.bnmax11 = nn.BatchNorm1d(64)
        self.bnmax12 = nn.BatchNorm1d(64)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),  # 3*64=384
                                   self.bn1,  # 2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))  # 0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),  # 64*64=4096
                                   self.bn2,  # 256
                                   nn.LeakyReLU(negative_slope=0.2))  # 0
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),  # 128*64=8096
                                   self.bn3,  # 256
                                   nn.LeakyReLU(negative_slope=0.2))  # 0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),  # 64*64=4096
                                   self.bn4,  # 256
                                   nn.LeakyReLU(negative_slope=0.2))  # 0
        self.bn__ = nn.BatchNorm1d(1024)
        self.conv__ = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),  # 128*64=8096
                                    self.bn__,  # 256
                                    nn.LeakyReLU(negative_slope=0.2))  # 0
        self.bn5 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(1024 + 512, 512, 1)
        self.dp5 = nn.Dropout(0.5)

        self.bn6 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.conv7 = nn.Conv1d(256, self.args.num_segmentation_type, 1)
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn9 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn10 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, 5)

    def forward(self, x, input_for_alignment_all_structure):  # input (B,3,N)
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.float()

        trans = self.s3n(x)  # (B,3,N)->(B,3,3)
        x = x.permute(0, 2, 1)  # (B, 3, N)->(B, N,3)
        x = torch.bmm(x, trans)  # (B,N,3)*(B,3,3)->(B,N,3)
        # Visuell_PointCloud_per_batch(x,target)
        x = x.permute(0, 2, 1)  # (B,N,3)->(B,3,N)

        x = get_neighbors(x, k=self.k)  # (B, 3, N) -> (B, 3*2, N, k)
        x = self.conv1(x)  # (B, 3*2, N, k) -> (B, 64, N, k)
        x = self.conv2(x)  # (B, 64, N, k) -> (B, 64, N, k)
        x1, _ = x.max(dim=-1, keepdim=False)  # (B, 64, N, k) -> (B, 64, N)

        x = get_neighbors(x1, k=self.k)  # (B, 64, N) -> (B, 64*2, N, k)
        x = self.conv3(x)  # (B, 64*2, N, k) -> (B, 64, N, k)
        x = self.conv4(x)  # (B, 64, N, k) -> (B, 64, N, k)
        x2, _ = x.max(dim=-1, keepdim=False)  # (B, 64, N, k) -> (B, 64, N)

        x = torch.cat((x1, x2), dim=1)  # (B, 64, N, k)*2 ->(B, 128, N)

        x = x.permute(0, 2, 1)  # (B,128,N)->(B,N,128)
        x1 = self.sa1(x)  # (B,N,128)->(B,N,128)
        x2 = self.sa2(x1)  # (B,N,128)->(B,N,128)
        x3 = self.sa3(x2)  # (B,N,128)->(B,N,128)
        x4 = self.sa4(x3)  # (B,N,128)->(B,N,128)
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # (B,N,128)*4->(B,N,512)
        x = x.permute(0, 2, 1)  # (B,N,512)->(B,512,N)
        x__ = x
        x = self.conv__(x)  # (B, 512, N)->(B, 1024, N)
        x_class = x
        x, _ = x.max(dim=-1, keepdim=False)  # (B, 1024, N) -> (B, 1024)
        x = x.unsqueeze(-1).repeat(1, 1, num_points)  # (B, 1024)->(B, 1024, N)
        x = torch.cat((x, x__), dim=1)
        # (B,1024,N)+(B, 512,N) ->(B, 1536,N)
        x = self.relu(self.bn5(self.conv5(x)))  # (B, 1536,N)-> (B, 512,N)
        x = self.dp5(x)
        x = self.relu(self.bn6(self.conv6(x)))  # (B, 512,N) ->(B,256,N)
        x = self.conv7(x)  # # (B, 256,N) ->(B,num_segmentation_type,N)

        y1 = F.adaptive_max_pool1d(x_class, 1).view(batch_size, -1)  # (B,1024,N)->(B,1024)
        y2 = F.adaptive_avg_pool1d(x_class, 1).view(batch_size, -1)  # (B,1024,N)->(B,1024)
        y = torch.cat((y1, y2), 1)  # (B, 1024*2)

        y = F.leaky_relu(self.bn9(self.linear1(y)), negative_slope=0.2)  # (B, 1024*2) -> (B, 512)
        y = self.dp2(y)
        y = F.leaky_relu(self.bn10(self.linear2(y)), negative_slope=0.2)  # (B, 512) -> (B, 256)
        y = self.dp3(y)
        y = self.linear3(y)  # (B, 256) -> (B, 5)
        return x, trans, y, None
