"""PointMLP

Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework
Xu Ma and Can Qin and Haoxuan You and Haoxi Ran and Yun Fu

Reference:
https://github.com/ma-xu/pointMLP-pytorch
"""
import string
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import furthest_point_sample, random_sample, LocalAggregation, create_convblock2d, three_interpolate, \
    three_nn, gather_operation, create_linearblock, create_convblock1d, create_grouper
import logging
import copy
from ..build import MODELS
from ..layers import furthest_point_sample, fps
from ..layers.group import QueryAndGroup
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, sample_ratio, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.sample_ratio = sample_ratio
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz

        add_channel = 3 if self.use_xyz else 0
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

        self.samble_downsample = DownSampleToken(self.sample_ratio, channel + add_channel)

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = N // self.sample_ratio
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        '''
        fps_idx = furthest_point_sample(xyz, S).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]
        '''

        points, idx = self.samble_downsample(points)
        points = points.contiguous()
        new_xyz = index_points(xyz, idx)  # [B, npoint, 3]
        new_points = index_points(points, idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


@MODELS.register_module()
class PointMLPEncoderSamble(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs):
        super(PointMLPEncoder, self).__init__()
        self.stages = len(pre_blocks)
        self.embedding = ConvBNReLU1D(in_channels, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            # append local_grouper_list
            # local_grouper = create_grouper(group_args)
            local_grouper = LocalGrouper(last_channel, reduce, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
        self.out_channels = last_channel
        self.act = get_activation(activation)

    def forward(self, x, f0=None):

        return self.forward_cls_feat(x, f0)

    def forward_cls_feat(self, p, x=None):
        if isinstance(p, dict):
            p, x = p['pos'], p.get('x', None)
        if x is None:
            x = p.transpose(1, 2).contiguous()
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give p[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            p, x = self.local_grouper_list[i](p, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        return x


@MODELS.register_module()
class PointMLPSamble(PointMLPEncoder):
    def __init__(self, in_channels=3, num_classes=15, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], group_args=None, **kwargs):
        super().__init__(in_channels, embed_dim, groups, res_expansion, activation, bias, use_xyz,
                         normalize, dim_expansion, pre_blocks, pos_blocks, k_neighbors, reducers,
                         **kwargs
                         )
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, p, x=None):
        return self.forward_cls_feat(p, x)

    def forward_cls_feat(self, p, x=None):
        if hasattr(p, 'keys'):
            p, x = p['pos'], p.get('x', None)
        if x is None:
            x = p.transpose(1, 2).contiguous()
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give p[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            p, x = self.local_grouper_list[i](p, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


# -------- There is Point Mlp Original Model Config
def pointMLP(num_classes=40, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                           activation="relu", bias=False, use_xyz=False, normalize="anchor",
                           dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                           k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pointMLPElite(num_classes=40, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                           activation="relu", bias=False, use_xyz=False, normalize="anchor",
                           dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                           k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


class DownSampleToken(nn.Module):
    # def __init__(self, config_ds, layer):
    def __init__(self, stride, in_channels):
        super(DownSampleToken, self).__init__()

        self.stride = stride

        self.K = 32

        self.num_heads = 1
        self.bin_mode = 'token'
        self.relu_mean_order = 'mean_relu'

        self.num_bins = 6

        q_in = in_channels
        q_out = in_channels
        k_in = in_channels
        k_out = in_channels
        v_in = in_channels
        v_out = in_channels

        self.q_depth = int(q_out / self.num_heads)
        self.k_depth = int(k_out / self.num_heads)
        self.v_depth = int(v_out / self.num_heads)
        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)

        # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, self.num_bins))
        # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, self.num_bins)/torch.sqrt(q_in))
        self.bin_tokens = nn.Parameter(
            torch.normal(mean=0, std=1 / math.sqrt(q_in), size=(1, q_in, self.num_bins)))

        self.softmax = nn.Softmax(dim=-1)
        # downsample res link

        # bin
        self.bin_sample_mode = 'random'

        self.dynamic_boundaries = True
        self.bin_boundaries = None

        self.normalization_mode = "z_score"

        # boltzmann
        self.boltzmann_T = 0.1

        self.momentum_update_factor = 0.99

    def forward(self, x):
        # x: input points data, [B, N, C]
        x = x.transpose(1, 2)
        # x.shape == (B, C, N)

        B, C, N = x.shape
        if self.bin_mode == 'token':
            bin_tokens = einops.repeat(self.bin_tokens, '1 c num_bins -> b c num_bins', b=B)
            # bin_tokens.shape ==(B,C,num_bins)
            x_and_token = torch.concat((x, bin_tokens), dim=2)  # x: (B,C,N+num_bins)

            q = self.q_conv(x)
            # q.shape == (B, C, N)
            q = self.split_heads(q, self.num_heads, self.q_depth)
            # q.shape == (B, H, D, N)
            q = q.permute(0, 1, 3, 2)  # q.shape == (B, H, N, D)

            k = self.k_conv(x_and_token)
            # k.shape ==  (B, C, N+num_bins)
            k = self.split_heads(k, self.num_heads, self.k_depth)
            # k.shape == (B, H, D, N+num_bins)
            v = self.v_conv(x_and_token)
            # v.shape ==  (B, C, N+num_bins)
            v = self.split_heads(v, self.num_heads, self.v_depth)
            # v.shape == (B, H, D, N+num_bins)

            energy = q @ k  # energy.shape == (B, H, N, N+num_bins)

            scale_factor = math.sqrt(q.shape[-1])

            attention_map_beforesoftmax = energy / scale_factor

            attention_map = self.softmax(attention_map_beforesoftmax)  # attention.shape == (B, H, N, N+num_bins)

            _, attention_bins_beforesoftmax = torch.split(attention_map_beforesoftmax, N, dim=-1)
            # attention_bins_beforesoftmax: (B,1,N,num_bins)
            attention_points, attention_bins = torch.split(attention_map, N, dim=-1)

        else:
            raise NotImplementedError

        self.attention_point_score, _, _ = self.calculate_attention_score(x, attention_points)
        # self.attention_point_score: (B, H, N)

        self.bin_boundaries, self.bin_points_mask = bin_partition(self.attention_point_score,
                                                                  self.bin_boundaries,
                                                                  self.dynamic_boundaries,
                                                                  self.momentum_update_factor,
                                                                  self.normalization_mode,
                                                                  self.num_bins)
        # self.bin_points_mask: (B,H,N,num_bins)
        # normalized_attention_point_score: (B,H,N)

        bin_weights, self.bin_weights_beforerelu = self.bin_weghts_calculation(attention_bins_beforesoftmax,
                                                                               self.bin_points_mask,
                                                                               self.relu_mean_order)

        # self.bin_points_mask: (B, H, N, num_bins)
        max_num_points = torch.sum(self.bin_points_mask.squeeze(dim=1), dim=1)
        # max_num_points:(B,num_bins)
        self.k_point_to_choose = calculate_num_points_to_choose(bin_weights, max_num_points, self.stride)
        # k_point_to_choose.shape == (B, num_bins)

        # attention_point_score = (self.attention_point_score - torch.mean(self.attention_point_score, dim=2, keepdim=True)) \
        #                         / torch.std(self.attention_point_score, dim=2, unbiased=False, keepdim=True)
        # import pickle
        # data_dict = {}
        # masked_attention_score = attention_point_score.unsqueeze(3) * bin_points_mask
        # data_dict["masked_attention_score"] = masked_attention_score
        # with open(f'/home/ies/fu/train_output/masked_attention_score.pkl', 'wb') as f:
        #     pickle.dump(data_dict, f)
        #     print('saved')

        index_down = generating_downsampled_index(
            self.attention_point_score,
            self.bin_points_mask,
            self.bin_sample_mode,
            self.boltzmann_T,
            self.k_point_to_choose)

        # attention_down = torch.gather(attention_map, dim=2,
        #                               index=index_down.unsqueeze(3).expand(-1, -1, -1, attention_map.shape[-1]))
        # attention_down.shape == (B, H, M, N+num_bins)
        # v_down = (attention_down @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        v_down = (attention_map @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # attention_down: (B, H, M, N+num_bins)
        # v.shape == (B, H, D, N+num_bins)
        # v_down.shape == (B, M, H, D)
        f = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)
        # f.shape: B,C,N
        # residual & feedforward

        f = f.transpose(1, 2)
        # f: output points data, [B, N, C]

        # index_down: sample index data, [B, M]
        return f, index_down.squeeze(1)
        # return (x_ds, index_down), (None, None)

    def bin_weghts_calculation(self, attention_bins_beforesoftmax, bin_points_mask, relu_mean_order):
        masked_attention_map_token = attention_bins_beforesoftmax * bin_points_mask
        if relu_mean_order == 'mean_relu':

            bin_weights_beforerelu = torch.sum(masked_attention_map_token, dim=2) / (
                    torch.count_nonzero(bin_points_mask, dim=2) + 1e-8)
            # torch.count_nonzero(masked_attention_map_token, dim=2) + 1e-8)
            bin_weights_beforerelu = bin_weights_beforerelu.squeeze(1)
            bin_weights = F.relu(bin_weights_beforerelu)
        elif relu_mean_order == 'relu_mean':
            masked_attention_map_token = F.relu(masked_attention_map_token)
            bin_weights_beforerelu = torch.sum(masked_attention_map_token, dim=2) / (
                    torch.count_nonzero(bin_points_mask, dim=2) + 1e-8)
            bin_weights_beforerelu = bin_weights_beforerelu.squeeze(1)
            bin_weights = bin_weights_beforerelu
        else:
            raise NotImplementedError
        return bin_weights, bin_weights_beforerelu

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x

    def res_block(self, x, x_ds, idx):  # x.shape == (B, C, N), x_ds.shape == (B, C, M)
        x_tmp = torch.gather(x, dim=-1, index=idx)  # x_res.shape == (B, 1, M)
        x_res = self.bn1(x_ds + x_tmp)  # x_res.shape == (B, C, M)
        if self.ff == True:
            x_tmp = self.ffn(x_res)
            x_res = self.bn2(x_ds + x_tmp)
        return x_res  # x_res.shape == (B, C, M)

    def get_sparse_attention_map(self, x, attention_points):
        mask = neighbor_mask(x, self.K)
        mask = mask.unsqueeze(1).expand(-1, attention_points.shape[1], -1, -1)
        # print(f'attention_map.shape{self.attention_map.shape}')
        # print(f'mask.shape{mask.shape}')
        # exit(-1)
        sparse_attention_map = attention_points * mask
        return mask, sparse_attention_map

    def calculate_attention_score(self, x, attention_points):
        mask, sparse_attention_map = self.get_sparse_attention_map(x, attention_points)
        sparse_num = torch.sum(mask, dim=-2) + 1e-8
        # sparse_num = torch.sum(mask, dim=-2) + 1

        # full attention map based
        attention_point_score = torch.sum(sparse_attention_map, dim=-2) / sparse_num / sparse_num

        attention_point_score[torch.isnan(attention_point_score)] = 0

        return attention_point_score, sparse_attention_map, mask


def calculate_num_points_to_choose(bin_prob, max_num_points, stride):
    """

    :param total_points_to_choose: Int
    :param bin_prob: torch.Tensor(B,num_bins)
    :param max_num_points: torch.Tensor(B,num_bins)
    :return: number of choosen points, torch.Tensor(B,num_bins)
    """
    total_points_to_choose = torch.sum(max_num_points[0, :]) // stride  # max_num_points // stride

    # print(f'max_num_points:{max_num_points}')
    # print(f'bin_prob:{bin_prob}')
    B, num_bins = bin_prob.shape
    bin_prob = bin_prob * max_num_points
    bin_prob += 1e-10

    # print(f'bin_prob:{bin_prob}')
    # print(f'max_num_points:{max_num_points}')

    num_chosen_points_in_bin = torch.zeros_like(bin_prob, device=bin_prob.device)
    for _ in range(num_bins):
        bin_prob = bin_prob / torch.sum(bin_prob, dim=1, keepdim=True)
        num_to_choose = total_points_to_choose - torch.sum(num_chosen_points_in_bin, dim=1, keepdim=True)

        if torch.all(num_to_choose == 0):
            break
        # print(torch.max(num_to_choose))

        # print(f'add:{bin_prob * num_to_choose}')
        num_chosen_points_in_bin += bin_prob * num_to_choose
        max_num_points = max_num_points.to(num_chosen_points_in_bin.dtype)
        num_chosen_points_in_bin = torch.where(num_chosen_points_in_bin >= max_num_points, max_num_points,
                                               num_chosen_points_in_bin)
        bin_prob = bin_prob * torch.where(num_chosen_points_in_bin >= max_num_points, 0, 1)

    num_chosen_points_in_bin = num_chosen_points_in_bin.int()
    # print(torch.argmax(max_num_points - num_chosen_points_in_bin, dim=1).shape)

    print(
        f"..........{num_chosen_points_in_bin[torch.arange(0, B), torch.argmax(max_num_points - num_chosen_points_in_bin, dim=1)].shape}")
    print(f'total_points_to_choose.shape:{total_points_to_choose.shape}')
    print(f'torch.sum(num_chosen_points_in_bin, dim=1).shape:{torch.sum(num_chosen_points_in_bin, dim=1).shape}')
    num_chosen_points_in_bin[
        torch.arange(0, B), torch.argmax(max_num_points - num_chosen_points_in_bin,
                                         dim=1)] += total_points_to_choose - torch.sum(num_chosen_points_in_bin, dim=1)

    # if torch.min(num_chosen_points_in_bin) < 0:
    #     for i in range(B):
    #         num_chosen_points_in_bin_one_batch = num_chosen_points_in_bin[i, :]
    #         if torch.min(num_chosen_points_in_bin_one_batch) < 0:
    #             min = torch.min(num_chosen_points_in_bin_one_batch)
    #             num_chosen_points_in_bin[i, torch.argmin(num_chosen_points_in_bin_one_batch)] -= min
    #             num_chosen_points_in_bin[i, torch.argmax(num_chosen_points_in_bin_one_batch)] += min

    # print(num_chosen_points_in_bin)
    # print(torch.sum(num_chosen_points_in_bin, dim=1))
    # print(max_num_points)
    # print(f'num_chosen_points_in_bin:{num_chosen_points_in_bin}')
    return num_chosen_points_in_bin


def generating_downsampled_index(attention_point_score, bin_points_mask, bin_sample_mode, boltzmann_t,
                                 k_point_to_choose):
    M = torch.sum(k_point_to_choose[0, :])

    B, _, N, num_bins = bin_points_mask.shape
    if bin_sample_mode == "topk":
        # attention_point_score: (B, H, N)
        attention_point_score = attention_point_score + 1e-8

        # bin_points_mask: (B, H, N, num_bins)
        masked_attention_point_score = attention_point_score.unsqueeze(3) * bin_points_mask
        # masked_attention_point_score: (B, H, N, num_bins)

        _, attention_index_score = torch.sort(masked_attention_point_score, dim=2, descending=True)
        attention_index_score = attention_index_score.squeeze(dim=1)
        # attention_index_score: (B, N, num_bins)

        index_down = []
        for batch_index in range(B):
            sampled_index_in_one_batch = []
            for bin_index in range(num_bins):
                sampled_index_in_one_batch.append(
                    attention_index_score[batch_index, :k_point_to_choose[batch_index, bin_index], bin_index])
            index_down.append(torch.concat(sampled_index_in_one_batch))
        index_down = torch.stack(index_down).reshape(B, 1, -1)
        # sampled_index: (B,H,M)

    elif bin_sample_mode == "uniform" or bin_sample_mode == "random":

        if bin_sample_mode == "uniform":
            # bin_points_mask: (B, H, N, num_bins)
            sampling_probabilities = bin_points_mask.float().squeeze(dim=1)

            sampling_probabilities = \
                sampling_probabilities + (torch.sum(sampling_probabilities, dim=1, keepdim=True) == 0)

        elif bin_sample_mode == "random":
            attention_point_score = (attention_point_score - torch.mean(attention_point_score, dim=2, keepdim=True)) \
                                    / torch.std(attention_point_score, dim=2, unbiased=False, keepdim=True)
            attention_point_score = torch.nn.functional.tanh(attention_point_score)
            # attention_point_score: (B, H, N)

            boltzmann_t_inverse = 1 / boltzmann_t

            # sampling_probabilities = torch.exp(attention_point_score.unsqueeze(3) / boltzmann_t) * bin_points_mask
            sampling_probabilities = torch.exp(
                attention_point_score.unsqueeze(3) * boltzmann_t_inverse) * bin_points_mask
            # sampling_probabilities = torch.exp(attention_point_score.unsqueeze(3) / 0.01) * bin_points_mask
            sampling_probabilities = sampling_probabilities / torch.sum(sampling_probabilities, dim=2, keepdim=True)

            # sampling_probabilities_np = sampling_probabilities.permute(0,1,3,2).cpu().numpy()
            # std_np = np.zeros((6,))
            # maxvalue = np.zeros((6,))
            # minvalue = np.zeros((6,))
            # meanvalue = np.zeros((6,))
            # for x in range(6):
            #
            #     sampling_probabilities_np_0 = sampling_probabilities_np[0, 0,  x,:]
            #     sampling_probabilities_np_0 = sampling_probabilities_np_0[sampling_probabilities_np_0 != 0]
            #
            #     maxvalue[x] = np.max(sampling_probabilities_np_0)
            #     minvalue[x] = np.min(sampling_probabilities_np_0)
            #     meanvalue[x] = np.mean(sampling_probabilities_np_0)
            #     std_np[x] = np.std(sampling_probabilities_np_0)
            # #
            # std_np0 = np.zeros((6,))
            # maxvalue0 = np.zeros((6,))
            # minvalue0 = np.zeros((6,))
            # meanvalue0 = np.zeros((6,))
            # for x in range(6):
            #     maxvalue0[x] = np.max(attention_point_score_np[ x,0,:])
            #     minvalue0[x] = np.min(attention_point_score_np[ x,0,:])
            #     meanvalue0[x] = np.mean(attention_point_score_np[ x,0,:])
            #     std_np0[x] = np.std(attention_point_score_np[ x,0,:])

            sampling_probabilities = sampling_probabilities.squeeze(dim=1)
            # sampling_probabilities: (B,N,num_bins)

            sampling_probabilities[torch.isnan(sampling_probabilities)] = 1e-8

        sampling_probabilities = sampling_probabilities.permute(0, 2, 1).reshape(-1, N)
        # sampling_probabilities: (B*num_bins,N)

        sampled_index_M_points = torch.multinomial(sampling_probabilities, M)
        # sampled_index_M_points: (B*num_bins,M)
        sampled_index_M_points = sampled_index_M_points.reshape(B, num_bins, M)
        # sampled_index_M_points: (B,num_bins,M)

        index_down = []
        for batch_index in range(B):
            sampled_index_in_one_batch = []
            for bin_index in range(num_bins):
                sampled_index_in_one_batch.append(
                    sampled_index_M_points[batch_index, bin_index, :k_point_to_choose[batch_index, bin_index]])
            index_down.append(torch.concat(sampled_index_in_one_batch))
        index_down = torch.stack(index_down).reshape(B, 1, -1)
        # sampled_index: (B,H,M)

    else:
        raise ValueError(
            'Please check the setting of bin sample mode. It must be topk, multinomial or random!')
    return index_down


def neighbor_mask(pcd, K):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    _, idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    B, N, _ = idx.shape
    mask = torch.zeros(B, N, N, dtype=torch.float32, device=idx.device)  # mask.shape == (B, N, N)
    mask.scatter_(2, idx, 1.0)
    return mask


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    a_mean = torch.mean(a, dim=1, keepdim=True)
    a = a - a_mean
    b = b - a_mean

    a_std = torch.mean(torch.std(a, dim=1, keepdim=True), dim=2, keepdim=True)
    a = a / a_std
    b = b / a_std

    # inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    # aa = torch.sum(a ** 2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    # bb = torch.sum(b ** 2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    # pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    pairwise_distance = -torch.cdist(a, b)  # , compute_mode='donot_use_mm_for_euclid_dist')

    # diff = torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=2)
    # pairwise_distance = torch.sum(diff ** 2, dim=-1)
    # num_positive = torch.sum(pairwise_distance > 0)

    distance, idx = pairwise_distance.topk(k=k, dim=-1)  # idx.shape == (B, N, K)
    return distance, idx


def bin_partition(attention_point_score, bin_boundaries, dynamic_boundaries_enable, momentum_update_factor,
                  normalization_mode, num_bins):
    B, H, N = attention_point_score.shape

    if bin_boundaries is not None:
        bin_boundaries = [item.to(attention_point_score.device) for item in bin_boundaries]

    # print(f'B{B},H{H},N{N}')
    # bin_boundaries = [item.to(attention_point_score.device) for item in bin_boundaries]
    if normalization_mode == 'no_normalization':
        pass
    elif normalization_mode == 'z_score':
        # attention_point_score: (B,1,N)
        attention_point_score = (attention_point_score - torch.mean(attention_point_score, dim=2, keepdim=True)) \
                                / torch.std(attention_point_score, dim=2, unbiased=False, keepdim=True)
    elif normalization_mode == 'z_score_no_std':
        attention_point_score = torch.log(attention_point_score)
        # try:
        #     attention_point_score = torch.log(attention_point_score)
        # except:
        #     print(f'----------Error in log-----------------')
        #     print(f'attention_point_score:\n{attention_point_score}')
        #     print(f'zero or negative value exists = {torch.min(attention_point_score).item() <= 0}')
        #     print(f'minimun is {torch.min(attention_point_score).item()}')

        # attention_point_score = attention_point_score - torch.mean(attention_point_score, dim=2, keepdim=True)
        attention_point_score_no_infnan = torch.where((attention_point_score == float('-inf')) |
                                                      (attention_point_score == float('inf')) |
                                                      torch.isnan(attention_point_score), 0, attention_point_score)
        attention_point_score = attention_point_score - torch.mean(attention_point_score_no_infnan, dim=2, keepdim=True)
        attention_point_score = torch.where((attention_point_score == float('inf')), 100, attention_point_score)
        attention_point_score = torch.where(torch.isnan(attention_point_score), 0, attention_point_score)
    else:
        raise NotImplementedError

    attention_point_score = attention_point_score.reshape(B, H, N, 1)
    # bin_boundaries: [(1,1,1,6),(1,1,1,6)]
    if dynamic_boundaries_enable:
        bin_boundaries = update_sampling_score_bin_boundary(bin_boundaries, attention_point_score, num_bins,
                                                            momentum_update_factor)
    bin_points_mask = (attention_point_score < bin_boundaries[0]) & (attention_point_score >= bin_boundaries[1])
    # bin_points_mask: (B,H,N,num_bins)
    return bin_boundaries, bin_points_mask


def update_sampling_score_bin_boundary(old_bin_boundaries, attention_point_score, num_bins, momentum_update_factor):
    # old_bin_boundaries:2 * (1,1,1,num_bins)
    # attention_point_score: (B, H, N)

    num_sampling_scores = attention_point_score.nelement()

    bin_boundaries_index = torch.arange(1, num_bins) / num_bins * num_sampling_scores
    bin_boundaries_index = bin_boundaries_index.to(attention_point_score.device).to(torch.int64)

    sorted_scores, _ = torch.sort(attention_point_score.flatten(), dim=0, descending=True)
    # print(bin_boundaries_index)
    bin_boundaries = sorted_scores[bin_boundaries_index]

    try:
        world_size = torch.distributed.get_world_size()
    except Exception as e:
        pass
    else:
        torch.distributed.all_reduce(bin_boundaries)  # , reduce_op=torch.distributed.ReduceOp.SUM)
        bin_boundaries = bin_boundaries / world_size

    if old_bin_boundaries is not None:
        new_bin_boundaries = [old_bin_boundaries[0].detach(), old_bin_boundaries[1].detach()]

        bin_boundaries = new_bin_boundaries[0][0, 0, 0, 1:] * momentum_update_factor + (
                1 - momentum_update_factor) * bin_boundaries

        new_bin_boundaries[0][0, 0, 0, 1:] = bin_boundaries
        new_bin_boundaries[1][0, 0, 0, :-1] = bin_boundaries
    else:
        # self.bin_boundaries = config_ds.bin.bin_boundaries[layer]
        bin_boundaries_upper = torch.empty((num_bins,), device=attention_point_score.device)
        bin_boundaries_upper[0] = float('inf')
        bin_boundaries_upper[1:] = bin_boundaries

        bin_boundaries_lower = torch.empty((num_bins,), device=attention_point_score.device)
        bin_boundaries_lower[-1] = float('-inf')
        bin_boundaries_lower[:-1] = bin_boundaries

        new_bin_boundaries = [torch.tensor(bin_boundaries_upper).reshape(1, 1, 1, num_bins),
                              # [inf, 0.503, 0.031, -0.230, -0.427, -0.627]
                              torch.tensor(bin_boundaries_lower).reshape(1, 1, 1, num_bins)
                              # [0.503, 0.031, -0.230, -0.427, -0.627, -inf]
                              ]

        # print(f'new_bin_boundaries:{new_bin_boundaries}')
    return new_bin_boundaries
