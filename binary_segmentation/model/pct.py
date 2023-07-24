#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model.attention import SelfAttentionLayer
# from utilities.util import *
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


def knn(x, k):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        idx: sample index data, [B, N, K]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B,N,N)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def index_points_neighbors(x, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    batch_size = x.size(0)
    num_points = x.size(1)
    num_dims = x.size(2)

    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, -1, num_dims)

    return neighbors


def get_neighbors(x, k=20):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        feature_points: indexed points data, [B, 2*C, N, K]
    """
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    idx = knn(x, k)  # batch_size x num_points x 20
    x = x.transpose(2, 1).contiguous()
    neighbors = index_points_neighbors(x, idx)  # _    (B,N,C) -> (B,N,k,C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((neighbors - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        # init.constant_(self.transform.weight, 0)
        # init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # _                      (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # _                      (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # _    (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # _                      (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # _    (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        # _                                         (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)
        # _                                         (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # _                  (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # _           (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


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

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # bs features 2048
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PCTSeg(nn.Module):
    def __init__(self, args):
        super(PCTSeg, self).__init__()
        self.args = args
        self.k = args.model_para.k

        self.stn3d = TransformNet()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.sa1 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)
        self.sa2 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)
        self.sa3 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)
        self.sa4 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)

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

    def forward(self, x):
        num_points = x.size(2)
        x = x.float()  # _                              input (B,3,N)
        # transform_matrix = self.s3n(x)
        transform_matrix = self.stn3d(get_neighbors(x, k=self.k))
        # _                                             (B,3,N) -> (B,3,3)
        x = x.permute(0, 2, 1)  # _                     (B,3,N) -> (B,N,3)
        x = torch.bmm(x, transform_matrix)  # _         (B,N,3) * (B,3,3) -> (B,N,3)
        # Visuell_PointCloud_per_batch(x,target)
        x = x.permute(0, 2, 1)  # _                     (B,N,3) -> (B,3,N)

        x = get_neighbors(x, k=self.k)  # _             (B,3,N) -> (B,3*2,N,k)
        x = self.conv1(x)  # _                          (B,3*2,N,k) -> (B,64,N,k)
        x = self.conv2(x)  # _                          (B,64,N,k) -> (B,64,N,k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # _       (B,64,N,k) -> (B,64,N)

        x = get_neighbors(x1, k=self.k)  # _            (B,64,N) -> (B,64*2,N,k)
        x = self.conv3(x)  # _                          (B,64*2,N,k) -> (B,64,N,k)
        x = self.conv4(x)  # _                          (B,64,N,k) -> (B,64,N,k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # _       (B,64,N,k) -> (B,64,N)

        x = torch.cat((x1, x2), dim=1)  # _             (B,64,N)*2 -> (B,128,N)

        x = x.permute(0, 2, 1)  # _                     (B,128,N) -> (B,N,128)
        x1 = self.sa1(x)  # _                           (B,N,128) -> (B,N,128)
        x2 = self.sa2(x1)  # _                          (B,N,128) -> (B,N,128)
        x3 = self.sa3(x2)  # _                          (B,N,128) -> (B,N,128)
        x4 = self.sa4(x3)  # _                          (B,N,128) -> (B,N,128)
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # _    (B,N,128)*4 -> (B,N,512)
        x = x.permute(0, 2, 1)  # _                     (B,N,512) -> (B,512,N)
        x__ = x
        x = self.conv__(x)  # _                         (B,512,N) -> (B,1024,N)
        x = x.max(dim=-1, keepdim=False)[0]  # _        (B,1024,N) -> (B,1024)
        x = x.unsqueeze(-1).repeat(1, 1, num_points)
        # _                                             (B,1024) -> (B,1024,N)
        x = torch.cat((x, x__), dim=1)
        # _                                             (B,1024,N) + (B,512,N) -> (B,1536,N)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)  # _     (B,1536,N) -> (B,512,N)
        x = self.dp5(x)  # _                            (B,512,N) -> (B,512,N)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)  # _     (B,512,N) -> (B,256,N)
        segmentation_labels = self.conv7(x)  # _        (B,256,N) -> (B,segment_type,N)

        return segmentation_labels, transform_matrix
