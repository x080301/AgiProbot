#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import einops
import numpy as np

# from utilities.util import *
from models.attention import SelfAttentionLayer


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


class InputEmbedding(nn.Module):
    def __init__(self, args):
        super(InputEmbedding, self).__init__()
        self.k = args.model_para.k

        self.stn3d = TransformNet()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

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

    def forward(self, x):
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

        return x, transform_matrix


class SegmentationMLP(nn.Module):
    def __init__(self, num_segmentation_type):
        super(SegmentationMLP, self).__init__()

        self.bn4 = nn.BatchNorm1d(1024)
        self.conv4 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)

        self.bn5 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(1024 + 512, 512, kernel_size=1)
        self.dp5 = nn.Dropout(0.5)

        self.bn6 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv7 = nn.Conv1d(256, num_segmentation_type, 1)

    def forward(self, x):
        _, _, num_points = x.shape  # _                 input (B,512,N)

        residual = x
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        # _                                             (B,512,N) -> (B,1024,N)
        x = x.max(dim=-1, keepdim=False)[0]  # _        (B,1024,N) -> (B,1024)

        x = einops.repeat(x, 'b c -> b c n', n=num_points)  # x = x.unsqueeze(-1).repeat(1, 1, num_points)
        # _                                             (B,1024) -> (B,1024,N)
        x = torch.cat((x, residual), dim=1)
        # _                                             (B,1024,N) + (B,512,N) -> (B,1536,N)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        # _                                             (B,1536,N) -> (B,512,N)
        x = self.dp5(x)  # _                            (B,512,N) -> (B,512,N)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        # _                                             (B,512,N) -> (B,256,N)
        segmentation_labels = self.conv7(x)  # _        (B,256,N) -> (B,segment_type,N)

        return segmentation_labels


class BoltTokenMLP(nn.Module):
    def __init__(self, args):
        super(BoltTokenMLP, self).__init__()
        self.args = args

        if self.args.model_para.token.num_mlp == 1:
            if self.args.model_para.token.mlp_dropout == 0:

                self.linear1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn1 = nn.BatchNorm1d(256)

                self.linear2 = nn.Conv1d(256, self.args.model_para.token.bolt_type + 2 + 3 + 3, kernel_size=1)
            else:
                self.linear1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn1 = nn.BatchNorm1d(256)
                self.dp = nn.Dropout(0.5)

                self.linear2 = nn.Conv1d(256, self.args.model_para.token.bolt_type + 2 + 3 + 3, kernel_size=1)
        elif self.args.model_para.token.num_mlp == 3:
            if self.args.model_para.token.mlp_dropout == 0:
                # existing label
                self.linear_existing1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn_existing1 = nn.BatchNorm1d(256)

                self.linear_existing2 = nn.Conv1d(256, 2, kernel_size=1)

                # type_label
                self.linear_type1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn_type1 = nn.BatchNorm1d(256)

                self.linear_type2 = nn.Conv1d(256, self.args.model_para.token.bolt_type, kernel_size=1)

                # center position of bolts
                self.linear_center1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn_center1 = nn.BatchNorm1d(256)

                self.linear_center2 = nn.Conv1d(256, 3, kernel_size=1)

                # normal of bolts
                self.linear_normal1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn_normal1 = nn.BatchNorm1d(256)

                self.linear_normal2 = nn.Conv1d(256, 3, kernel_size=1)

            else:
                # existing label
                self.linear_existing1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn_existing1 = nn.BatchNorm1d(256)
                self.dp_existing = nn.Dropout(0.5)

                self.linear_existing2 = nn.Conv1d(256, 2, kernel_size=1)

                # type_label
                self.linear_type1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn_type1 = nn.BatchNorm1d(256)
                self.dp_type = nn.Dropout(0.5)

                self.linear_type2 = nn.Conv1d(256, self.args.model_para.token.bolt_type, kernel_size=1)

                # center position of bolts
                self.linear_center1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn_center1 = nn.BatchNorm1d(256)

                self.linear_center2 = nn.Conv1d(256, 3, kernel_size=1)

                # normal of bolts
                self.linear_normal1 = nn.Conv1d(512, 256, kernel_size=1)
                self.bn_normal1 = nn.BatchNorm1d(256)

                self.linear_normal2 = nn.Conv1d(256, 3, kernel_size=1)

    def forward(self, x):
        # _                                                     input (B,512,T)

        if self.args.model_para.token.num_mlp == 1:
            if self.args.model_para.token.mlp_dropout == 0:

                x = F.leaky_relu(self.self.bn1(self.linear1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)

                x = self.linear2(x)  # _                        (B,256,T) -> (B,bolt_type+2+3+3,T)

                existing_label, type_labels, bolt_centers, bolt_normals = \
                    torch.split(x, [2, self.args.model_para.token.bolt_type, 3, 3], dim=1)
                # _                                             (B,bolt_type+2+3+3,T) ->
                # _                                                       (B,2,T),(B,bolt_type,T),(B,3,T),(B,3,T)

            else:
                x = F.leaky_relu(self.self.bn1(self.linear1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)
                x = self.dp(x)  # _                             (B,256,T) -> (B,256,T)

                x = self.linear2(x)  # _                        (B,256,T) -> (B,bolt_type+2+3+3,T)

                existing_label, type_labels, bolt_centers, bolt_normals = \
                    torch.split(x, [2, self.args.model_para.token.bolt_type, 3, 3], dim=1)
                # _                                             (B,bolt_type+2+3+3,T) ->
                # _                                                       (B,2,T),(B,bolt_type,T),(B,3,T),(B,3,T)

        else:  # elif self.args.model_para.token.num_mlp == 4:
            if self.args.model_para.token.mlp_dropout == 0:

                # existing label
                existing_label = F.leaky_relu(self.self.bn_existing1(self.linear_existing1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)

                existing_label = self.linear_existing2(existing_label)
                # _                                             (B,256,T) -> (B,2,T)

                # type_label
                type_labels = F.leaky_relu(self.self.bn_type1(self.linear_type1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)

                type_labels = self.linear_type2(type_labels)
                # _                                             (B,256,T) -> (B,bolt_type,T)

                # center position of bolts
                bolt_centers = F.leaky_relu(self.self.bn_center1(self.linear_center1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)

                bolt_centers = self.linear_center2(bolt_centers)
                # _                                             (B,256,T) -> (B,3,T)

                # normal of bolts
                bolt_normals = F.leaky_relu(self.self.bn_normal1(self.linear_normal1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)

                bolt_normals = self.linear_normal2(bolt_normals)
                # _                                             (B,256,T) -> (B,3,T)

            else:

                # existing label
                existing_label = F.leaky_relu(self.self.bn_existing1(self.linear_existing1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)
                x = self.dp_existing(x)  # _                    (B,256,T) -> (B,256,T)

                existing_label = self.linear_existing2(existing_label)
                # _                                             (B,256,T) -> (B,2,T)

                # type_label
                type_labels = F.leaky_relu(self.self.bn_type1(self.linear_type1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)
                x = self.dp_type(x)  # _                        (B,256,T) -> (B,256,T)

                type_labels = self.linear_type2(type_labels)
                # _                                             (B,256,T) -> (B,bolt_type,T)

                # center position of bolts
                bolt_centers = F.leaky_relu(self.self.bn_center1(self.linear_center1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)

                bolt_centers = self.linear_center2(bolt_centers)
                # _                                             (B,256,T) -> (B,3,T)

                # normal of bolts
                bolt_normals = F.leaky_relu(self.self.bn_normal1(self.linear_normal1(x)), negative_slope=0.2)
                # _                                             (B,512,T) -> (B,256,T)

                bolt_normals = self.linear_normal2(bolt_normals)
                # _                                             (B,256,T) -> (B,3,T)

        # existing_label = F.softmax(existing_label, dim=1)  # _  (B,2,T)
        # type_labels = F.softmax(type_labels, dim=1)  # _        (B,bolt_type,T)
        # bolt_centers = F.sigmoid(bolt_centers, dim=1)  # _      (B,3,T)
        # bolt_normals = bolt_normals  # _                        (B,3,T)

        return existing_label, type_labels, bolt_centers, bolt_normals


class PCTToken(nn.Module):

    def __init__(self, args):
        super(PCTToken, self).__init__()
        self.args = args

        self.bolt_token = nn.Parameter(torch.randn(1, 128, self.args.model_para.token.num_tokens))

        self.input_embedding = InputEmbedding(self.args)

        self.sa1 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)
        self.sa2 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)
        self.sa3 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)
        self.sa4 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)

        self.segmentation_mlp = SegmentationMLP(self.args.num_segmentation_type)
        self.bolt_token_mlp = BoltTokenMLP(self.args)

    def forward(self, x):
        batch_size, _, num_points = x.shape  # _        input (B,3,N)
        bolt_tokens = einops.repeat(self.bolt_token, '1 c t -> b c t', b=batch_size)
        # _                                             bolt_tokens (B,128,T)

        point_wise_features, transform_matrix = self.input_embedding(x)
        # _                                             (B,3,N) -> (B,128,N)

        x = torch.cat((point_wise_features, bolt_tokens), dim=-1)
        # _                                             (B,128,N) + (B,128,T) -> (B,128,N+T)

        x1 = self.sa1(x)  # _                           (B,128,N+T) -> (B,128,N+T)
        x2 = self.sa2(x1)  # _                          (B,128,N+T) -> (B,128,N+T)
        x3 = self.sa3(x2)  # _                          (B,128,N+T) -> (B,128,N+T)
        x4 = self.sa4(x3)  # _                          (B,128,N+T) -> (B,128,N+T)
        x = torch.cat((x1, x2, x3, x4), dim=-2)  # _    (B,128,N+T)*4 -> (B,512,N+T)

        point_wise_features, bolt_tokens = torch.split(x, [num_points, self.args.model_para.token.num_tokens], dim=-1)
        # _                                             (B,512,N+T) -> (B,512,N) + (B,512,T)

        point_segmentation_pred = self.segmentation_mlp(point_wise_features)
        # _                                             (B,512,N) -> (B,segment_type,N)

        bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals = self.bolt_token_mlp(bolt_tokens)
        # _                                             (B,512,T) -> (B,bolt_type+2+3+3,T)

        # return logits
        return point_segmentation_pred, bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals, transform_matrix


class PCTPipeline(nn.Module):

    def __init__(self, args):
        super(PCTPipeline, self).__init__()
        self.args = args

        self.input_embedding = InputEmbedding(self.args)

        self.sa1 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)
        self.sa2 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)
        self.sa3 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)
        self.sa4 = SelfAttentionLayer(in_channels=128, out_channels=128, num_heads=args.model_para.attentionhead)

        self.segmentation_mlp = SegmentationMLP(self.args.num_segmentation_type)

    def forward(self, x):
        batch_size, _, num_points = x.shape  # _        input (B,3,N)
        # _                                             bolt_tokens (B,128,T)

        point_wise_features, transform_matrix = self.input_embedding(x)
        # _                                             (B,3,N) -> (B,128,N)

        # _                                             (B,128,N) + (B,128,T) -> (B,128,N+T)

        x1 = self.sa1(point_wise_features)  # _         (B,128,N+T) -> (B,128,N+T)
        x2 = self.sa2(x1)  # _                          (B,128,N+T) -> (B,128,N+T)
        x3 = self.sa3(x2)  # _                          (B,128,N+T) -> (B,128,N+T)
        x4 = self.sa4(x3)  # _                          (B,128,N+T) -> (B,128,N+T)
        point_wise_features = torch.cat((x1, x2, x3, x4), dim=-2)
        # _                                             (B,128,N+T)*4 -> (B,512,N+T)

        # _                                             (B,512,N+T) -> (B,512,N) + (B,512,T)

        point_segmentation_pred = self.segmentation_mlp(point_wise_features)
        # _                                             (B,512,N) -> (B,segment_type,N)

        # return logits
        return point_segmentation_pred, transform_matrix

