#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: bixuelei
@Contact: xueleibi@gmail.com
@File: model.py
@Time: 2022/1/15 17:11 PM
"""
from util import *
from torch.autograd import Variable


# from display import *


def knn(x, k):
    """
    Input:
        points: input points data, [B, N, C]
    Return:
        idx: sample index data, [B, N, K]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
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
        feature_points:, indexed points data, [B, 2*C, N, K]
    """
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    idx = knn(x, k)  # batch_size x num_points x 20
    x = x.transpose(2, 1).contiguous()
    neighbors = index_points_neighbors(x, idx)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((neighbors - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


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


class ball_query_sample_with_goal(nn.Module):  # top1(ball) to approach
    def __init__(self, args, num_feats, input_dims, actv_fn=F.relu, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted

        """
        super(ball_query_sample_with_goal, self).__init__()
        self.point_after = args.after_stn_as_input
        self.args = args
        self.num_heads = args.num_heads
        self.num_layers = args.num_attention_layer
        self.num_latent_feats_inencoder = args.self_encoder_latent_features
        self.num_feats = num_feats
        self.actv_fn = actv_fn
        self.input_dims = input_dims

        self.top_k = 32
        self.d_model = 480
        self.radius = 0.3
        self.max_radius_points = 32

        self.self_atn_layer = SA_Layer_Multi_Head(args, 256)
        self.selfatn_layers = SA_Layers(self.num_layers, self.self_atn_layer)

        self.loss_function = nn.MSELoss()

        self.feat_channels_1d = [self.num_feats, 64, 64, 48]
        self.feat_generator = create_conv1d_serials(self.feat_channels_1d)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.feat_channels_1d[i + 1])
                for i in range(len(self.feat_channels_1d) - 1)
            ]
        )

        self.feat_channels_3d = [128, 256, self.d_model]
        self.radius_cnn = create_conv3d_serials(self.feat_channels_3d, self.max_radius_points, 3)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList(
            [
                nn.BatchNorm3d(num_features=self.feat_channels_3d[i])
                for i in range(len(self.feat_channels_3d))
            ]
        )

    def forward(self, hoch_features, input, x_a_r, target):  # [bs,1,features,n_points]   [bs,C,n_points]

        top_k = self.top_k  # 16
        origial_hoch_features = hoch_features  # [bs,1,features,n_points]
        feat_dim = input.shape[1]

        hoch_features_att = hoch_features
        #############################################################
        # implemented by myself
        #############################################################
        hoch_features_att = hoch_features_att.permute(0, 2, 1)
        hoch_features_att = self.selfatn_layers(hoch_features_att)
        hoch_features_att = hoch_features_att.permute(0, 2, 1)

        ##########################
        #
        ##########################
        high_inter = hoch_features_att
        for j, conv in enumerate(self.feat_generator):  # [bs,features,n_points]->[bs,10,n_points]
            bn = self.feat_bn[j]
            high_inter = self.actv_fn(bn(conv(high_inter)))
        topk = torch.topk(high_inter, k=top_k, dim=-1)  # [bs,10,n_points]->[bs,10,n_superpoint]
        indices_32 = topk.indices  # [bs,n_superpoints,top_k]->[bs,n_superpoints,top_k]
        # indices_bolts=indices_32[:,8:16,:]
        # visialize_cluster(input,indices_bolts)
        # indices=indices[:,:,0]
        # visialize_superpoints(input,indices)
        indices = indices_32[:, :, 0]
        result_net = torch.ones((1))
        if not self.args.test and self.args.training:  # [bs,n_cluster,top_k]->[bs,n_cluster,top_k]
            result_net = index_points(input.permute(0, 2, 1).float(), indices)

        sorted_input = torch.zeros((origial_hoch_features.shape[0], feat_dim, top_k)).to(
            input.device  # [bs,C,n_superpoint]
        )

        if top_k == 1:
            indices = indices.unsqueeze(dim=-1)

        sorted_input = index_points(input.permute(0, 2, 1).float(), indices).permute(0, 2,
                                                                                     1)  # [bs,n_superpoint]->[bs,C,n_superpoint]

        all_points = input.permute(0, 2, 1).float()  # [bs,C,n_points]->[bs,n_points,C]
        query_points = sorted_input.permute(0, 2, 1)  # [bs,C,n_superpoint]->[bs,n_superpoint,C]

        dis1 = square_distance(all_points, query_points)
        radius_indices = torch.topk(-dis1, k=32, dim=-2).indices.permute(0, 2, 1)
        # radius_indices = query_ball_point(                                       #idx=[bs,n_superpoint,n_sample]
        #     self.radius,
        #     self.max_radius_points,
        #     all_points[:, :, :3],
        #     query_points[:, :, :3],)

        if self.point_after:
            radius_points = index_points(x_a_r.permute(0, 2, 1), radius_indices)
        else:
            radius_points = index_points(all_points, radius_indices)  # [bs,n_superpoint,n_sample,C]
        radius_points = radius_points.unsqueeze(dim=1)

        for i, radius_conv in enumerate(self.radius_cnn):  # [bs,n_superpoint,n_sample+1,C]->[bs,512,n_superpoint,1,1]
            bn = self.radius_bn[i]
            radius_points = self.actv_fn(bn(radius_conv(radius_points)))

        radius_points = radius_points.squeeze(dim=-1).squeeze(
            dim=-1)  # [bs,512,n_superpoint,1,1]->[bs,512,n_superpoint]                                          #[bs,n_superpoint]
        radius_points = torch.cat((radius_points, indices_32.permute(0, 2, 1)), dim=1)
        return radius_points, result_net


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

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-6 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        x = x.permute(0, 2, 1)
        return x


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

    def forward(self, x, input_for_alignment_all_structure):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.float()

        transform_matrix = self.s3n(x)
        x = x.permute(0, 2, 1)  # (batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, transform_matrix)
        # Visuell_PointCloud_per_batch(x,target)
        x = x.permute(0, 2, 1)

        x = get_neighbors(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_neighbors(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2), dim=1)  # (batch_size, 64, num_points, k)*2 ->(batch_size, 128, num_points)

        x = x.permute(0, 2, 1)
        x1 = self.sa1(x)  # (batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)  50MB
        x2 = self.sa2(x1)  # (batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x3 = self.sa3(x2)  # (batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x4 = self.sa4(x3)  # (batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # (batch_size, 64*2, num_points)*4->(batch_size, 512, num_points)
        x = x.permute(0, 2, 1)
        x__ = x
        x = self.conv__(x)  # (batch_size, 512, num_points)->(batch_size, 1024, num_points)
        x_class = x
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = x.unsqueeze(-1).repeat(1, 1, num_points)  # (batch_size, 1024)->(batch_size, 1024, num_points)
        x = torch.cat((x, x__),
                      dim=1)  # (batch_size,2048,num_points)+(batch_size, 1024,num_points) ->(batch_size, 3036,num_points)
        x = self.relu(self.bn5(self.conv5(x)))  # (batch_size, 3036,num_points)-> (batch_size, 512,num_points)
        x = self.dp5(x)
        x = self.relu(self.bn6(self.conv6(x)))  # (batch_size, 512,num_points) ->(batch_size,256,num_points)
        segmentation_labels = self.conv7(x)  # # (batch_size, 256,num_points) ->(batch_size,6,num_points)

        return segmentation_labels, transform_matrix
