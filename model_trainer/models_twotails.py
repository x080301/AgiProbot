#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: bixuelei
@Contact: xueleibi@gmail.com
@File: model.py
@Time: 2022/1/15 17:11 PM
"""
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from util import SA_Layer_Single_Head

from torch.autograd import Variable
# from display import *


def knn(x, k):
    """
    Input:
        points: input points data, [B, N, C]
    Return:
        idx: sample index data, [B, N, K]
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)
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
    num_dims= x.size(2)

    device=idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx=idx+idx_base
    neighbors = x.view(batch_size*num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, -1, num_dims)

    return neighbors



def get_neighbors(x,k=20):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        feature_points:, indexed points data, [B, 2*C, N, K]
    """
    batch_size = x.size(0)
    num_dims= x.size(1)
    num_points = x.size(2)
    idx = knn(x, k)                                         # batch_size x num_points x 20
    x = x.transpose(2, 1).contiguous()
    neighbors = index_points_neighbors(x, idx)  
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) 
    feature = torch.cat((neighbors-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature



def visialize_cluster(input,indices):
        input=input.permute(0, 2, 1).float()
        bs_,n_point,_=input.shape
        to_display=input
        man_made_label=torch.zeros((bs_,n_point,1)).to(input.device)
        to_display=torch.cat((to_display,man_made_label),dim=-1)        #[bs,n_points,C+1]
        bs,n_superpoints,num_topk=indices.shape
        indices_=indices.view(bs,-1)
        sample_point=index_points(input,indices_)                       #from [bs,n_points,3] to[bs,n_superpoint*num_topk,3]
        sample_point=sample_point.view(bs,n_superpoints,num_topk,3)     #[bs,n_superpoint*num_topk,3]->[bs,n_superpoint,num_topk,3]
        man_made_points=torch.zeros((bs,n_superpoints,num_topk,4)).to(input.device)
        label_n_superpoints=torch.zeros((bs,n_superpoints,num_topk,1)).to(input.device)
        for i in range(n_superpoints):
            label_n_superpoints[:,i,:,:]=i+1
            man_made_points[:,i,:,0:3]=sample_point[:,i,:,0:3]
            man_made_points[:,i,:,3]=label_n_superpoints[:,i,:,0]
        man_made_points=man_made_points.view(bs,-1,4)                     
        for i in range(bs):
            sampled=man_made_points[i,:,:].squeeze(0)
            original=to_display[i,:,:].squeeze(0)
            Visuell_superpoint(sampled,original)            


def visialize_superpoints(input,indices):
        input=input.permute(0, 2, 1).float()
        bs_,n_point,_=input.shape
        to_display=input
        man_made_label=torch.zeros((bs_,n_point,1)).to(input.device)
        to_display=torch.cat((to_display,man_made_label),dim=-1)        #[bs,n_points,C+1]
        bs,n_superpoints=indices.shape
        num_topk=1
        indices_=indices.view(bs,-1)
        sample_point=index_points(input,indices_)                       #from [bs,n_points,3] to[bs,n_superpoint*num_topk,3]
        sample_point=sample_point.view(bs,n_superpoints,num_topk,3)     #[bs,n_superpoint*num_topk,3]->[bs,n_superpoint,num_topk,3]
        man_made_points=torch.zeros((bs,n_superpoints,num_topk,4)).to(input.device)
        label_n_superpoints=torch.zeros((bs,n_superpoints,num_topk,1)).to(input.device)
        for i in range(n_superpoints):
            label_n_superpoints[:,i,:,:]=i+1
            man_made_points[:,i,:,0:3]=sample_point[:,i,:,0:3]
            man_made_points[:,i,:,3]=label_n_superpoints[:,i,:,0]
        man_made_points=man_made_points.view(bs,-1,4)                     
        for i in range(bs):
            sampled=man_made_points[i,:,:].squeeze(0)
            original=to_display[i,:,:].squeeze(0)
            Visuell_superpoint(sampled,original)


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
        x = F.relu(self.bn1(self.conv1(x))) #bs features 2048
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


class PCT_core(nn.Module):
    def __init__(self, args):
        super(PCT_core, self).__init__()
        self.args = args
        self.k = args.k
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(1024)
        self.sa1=SA_Layer_Single_Head(128)
        self.sa2=SA_Layer_Single_Head(128)
        self.sa3=SA_Layer_Single_Head(128)
        self.sa4=SA_Layer_Single_Head(128)

                                               
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0


    def forward(self, x):
        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        #Visuell_PointCloud_per_batch(x,target)
        x=x.permute(0,2,1)

        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2), dim=1)   # (batch_size, 64, num_points, k)*2 ->(batch_size, 128, num_points)

        x=x.permute(0,2,1)
        x1 = self.sa1(x)                       #(batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)  50MB
        x2 = self.sa2(x1)                      #(batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x3 = self.sa3(x2)                      #(batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x4 = self.sa4(x3)                      #(batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x = torch.cat((x1, x2, x3, x4), dim=-1)      #(batch_size, 64*2, num_points)*4->(batch_size, 512, num_points)
        x_=x.permute(0,2,1)
        x = self.conv5(x_)                           # (batch_size, 512, num_points)->(batch_size, 1024, num_points) 
 
        return x,x_,trans


class TWO_CLS(nn.Module):
    def __init__(self,args):
        super(TWO_CLS, self).__init__()
        self.linear1 = nn.Linear(2048, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 5)
        
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.dp3 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dp4 = nn.Dropout(p=0.6)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)   # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)   # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)                              # (batch_size, emb_dims*2)
        
        ty = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)     # (batch_size, emb_dims) -> (batch_size, 512)
        ty = self.dp1(ty)
        ty = F.leaky_relu(self.bn2(self.linear2(ty)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)
        ty = self.dp2(ty)
        ty = self.linear3(ty)     # (batch_size, 256) -> (batch_size, 5)
        
        cover_existence = F.leaky_relu(self.bn3(self.fc1(x)), negative_slope=0.2)      # (batch_size, 1024*2) -> (batch_size, 512)
        cover_existence = self.dp3(cover_existence)
        cover_existence = F.leaky_relu(self.bn4(self.fc2(cover_existence)), negative_slope=0.2)      # (batch_size, 512) -> (batch_size, 256)
        cover_existence = self.dp4(cover_existence)
        cover_existence = self.fc3(cover_existence)         # (batch_size, 256) -> (batch_size, 3)
        return ty, cover_existence              # ty -> type cls, num -> num of cover bol


class PCT_semseg(nn.Module):
    def __init__(self, args):
        super(PCT_semseg, self).__init__()
        self.args = args      
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.linear1 = nn.Linear(5, 64, bias=False)
        self.linear2 = nn.Linear(2, 64)

        self.conv3 = nn.Sequential(nn.Conv1d(1024+512+128, 512, kernel_size=1, bias=False),     #3*64=384
                                   self.bn3,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))            
        self.conv7 = nn.Conv1d(256, self.args.num_segmentation_type, 1)
        self.dp1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        

    def forward(self, x,x_,x_calss,x_cover_existence):
        num_points = x.size(2)
        x_calss=F.leaky_relu(self.bn1(self.linear1(x_calss.float())), negative_slope=0.2)
        x_calss=x_calss.unsqueeze(-1).repeat(1,1,num_points)# (batch_size, 1024)->(batch_size, 1024, num_points)
        x_cover_existence=F.leaky_relu(self.bn2(self.linear2(x_cover_existence.float())), negative_slope=0.2)
        x_cover_existence=x_cover_existence.unsqueeze(-1).repeat(1,1,num_points)# (batch_size, 1024)->(batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]       # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x=x.unsqueeze(-1).repeat(1,1,num_points)# (batch_size, 1024)->(batch_size, 1024, num_points)
        x=torch.cat((x,x_,x_calss,x_cover_existence),dim=1)             # (batch_size,2048,num_points)+(batch_size, 1024,num_points) ->(batch_size, 3036,num_points)
        x=self.relu(self.bn3(self.conv3(x)))        # (batch_size, 3036,num_points)-> (batch_size, 512,num_points)
        x=self.dp1(x)                      
        x=self.relu(self.bn4(self.conv4(x)))        # (batch_size, 512,num_points) ->(batch_size,256,num_points)
        # self.dp1 = nn.Dropout(0.5)
        x=self.conv7(x)                             # # (batch_size, 256,num_points) ->(batch_size,6,num_points)
        return x