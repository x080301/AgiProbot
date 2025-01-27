import torch
from torch import nn
from utils import ops
from models import embedding
from models import attention
from models import downsample
from models import _downsample_token
import time
from utils.dataloader import fps
from models.fpsknn import PointNetSetAbstraction


class FeatureLearningBlock(nn.Module):
    def __init__(self, config_feature_learning_block):
        downsample_which = config_feature_learning_block.downsample.ds_which
        ff_conv2_channels_out = (
            config_feature_learning_block.attention.ff_conv2_channels_out
        )
        self.res_link_enable = config_feature_learning_block.res_link.enable
        fl_which = config_feature_learning_block.attention.fl_which

        super(FeatureLearningBlock, self).__init__()
        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(config_feature_learning_block.embedding, layer)
                for layer in range(len(config_feature_learning_block.embedding.K))
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSample(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleWithSigma(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "global_carve":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleCarve(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local_insert":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleInsert(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "token":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleToken(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        else:
            raise NotImplementedError
        if fl_which == "n2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Neighbor2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        elif fl_which == "p2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Point2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        else:
            raise ValueError("Only n2p and p2p are valid for fl_which!")

        if self.res_link_enable:
            self.conv_list = nn.ModuleList(
                [
                    nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False)
                    for channel_in in ff_conv2_channels_out
                ]
            )
            # self.conv_list = nn.ModuleList([nn.Sequential(nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False),
            #                                           nn.BatchNorm1d(1024),
            #                                           nn.LeakyReLU(negative_slope=0.2))
            #                             for channel_in in ff_conv2_channels_out])
        else:
            self.conv = nn.Conv1d(
                ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False
            )
            # self.conv = nn.Sequential(nn.Conv1d(ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False),
            #                           nn.BatchNorm1d(1024),
            #                           nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x_list = []
        x_xyz = x.clone()
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.feature_learning_layer_list[0](x)

        if self.res_link_enable:
            res_link_list = []
            res_link_list.append(self.conv_list[0](x).max(dim=-1)[0])
            for i in range(len(self.downsample_list)):
                (x, idx_select) = self.downsample_list[i](x, x_xyz)[0]
                x = self.feature_learning_layer_list[i + 1](x)
                x_xyz = ops.gather_by_idx(x_xyz, idx_select)
                res_link_list.append(self.conv_list[i + 1](x).max(dim=-1)[0])
            self.res_link_list = res_link_list
            x = torch.cat(res_link_list, dim=1)
            return x, res_link_list
        else:
            for i in range(len(self.downsample_list)):
                x = self.downsample_list[i](x)[0][0]
                x = self.feature_learning_layer_list[i + 1](x)
            x = self.conv(x).max(dim=-1)[0]
            return x


class FeatureLearningBlock_inference_time(nn.Module):
    def __init__(self, config_feature_learning_block):
        downsample_which = config_feature_learning_block.downsample.ds_which
        ff_conv2_channels_out = (
            config_feature_learning_block.attention.ff_conv2_channels_out
        )
        self.res_link_enable = config_feature_learning_block.res_link.enable
        fl_which = config_feature_learning_block.attention.fl_which

        super(FeatureLearningBlock_inference_time, self).__init__()
        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(config_feature_learning_block.embedding, layer)
                for layer in range(len(config_feature_learning_block.embedding.K))
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSample(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleWithSigma(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "global_carve":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleCarve(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local_insert":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleInsert(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "token":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleToken(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        else:
            raise NotImplementedError
        if fl_which == "n2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Neighbor2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        elif fl_which == "p2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Point2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        else:
            raise ValueError("Only n2p and p2p are valid for fl_which!")

        if self.res_link_enable:
            self.conv_list = nn.ModuleList(
                [
                    nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False)
                    for channel_in in ff_conv2_channels_out
                ]
            )
            # self.conv_list = nn.ModuleList([nn.Sequential(nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False),
            #                                           nn.BatchNorm1d(1024),
            #                                           nn.LeakyReLU(negative_slope=0.2))
            #                             for channel_in in ff_conv2_channels_out])
        else:
            self.conv = nn.Conv1d(
                ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False
            )
            # self.conv = nn.Sequential(nn.Conv1d(ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False),
            #                           nn.BatchNorm1d(1024),
            #                           nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x_list = []
        x_xyz = x.clone()
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.feature_learning_layer_list[0](x)

        inference_time_list = []
        if self.res_link_enable:
            res_link_list = []
            res_link_list.append(self.conv_list[0](x).max(dim=-1)[0])
            for i in range(len(self.downsample_list)):
                torch.cuda.synchronize()
                start_time = time.time()
                (x, idx_select) = self.downsample_list[i](x, x_xyz)[0]
                torch.cuda.synchronize()
                inference_time_list.append(time.time() - start_time)
                x = self.feature_learning_layer_list[i + 1](x)
                x_xyz = ops.gather_by_idx(x_xyz, idx_select)
                res_link_list.append(self.conv_list[i + 1](x).max(dim=-1)[0])
            self.res_link_list = res_link_list
            x = torch.cat(res_link_list, dim=1)
            return x, res_link_list, inference_time_list
        else:
            for i in range(len(self.downsample_list)):
                x = self.downsample_list[i](x)[0][0]
                x = self.feature_learning_layer_list[i + 1](x)
            x = self.conv(x).max(dim=-1)[0]
            return x


class FeatureLearningBlock_fps(nn.Module):
    def __init__(self, config_feature_learning_block):
        downsample_which = config_feature_learning_block.downsample.ds_which
        ff_conv2_channels_out = (
            config_feature_learning_block.attention.ff_conv2_channels_out
        )
        self.res_link_enable = config_feature_learning_block.res_link.enable
        fl_which = config_feature_learning_block.attention.fl_which

        super(FeatureLearningBlock_fps, self).__init__()
        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(config_feature_learning_block.embedding, layer)
                for layer in range(len(config_feature_learning_block.embedding.K))
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSample(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleWithSigma(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "global_carve":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleCarve(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local_insert":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleInsert(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "token":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleToken(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        else:
            raise NotImplementedError
        if fl_which == "n2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Neighbor2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        elif fl_which == "p2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Point2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        else:
            raise ValueError("Only n2p and p2p are valid for fl_which!")

        if self.res_link_enable:
            self.conv_list = nn.ModuleList(
                [
                    nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False)
                    for channel_in in ff_conv2_channels_out
                ]
            )
            # self.conv_list = nn.ModuleList([nn.Sequential(nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False),
            #                                           nn.BatchNorm1d(1024),
            #                                           nn.LeakyReLU(negative_slope=0.2))
            #                             for channel_in in ff_conv2_channels_out])
        else:
            self.conv = nn.Conv1d(
                ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False
            )
            # self.conv = nn.Sequential(nn.Conv1d(ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False),
            #                           nn.BatchNorm1d(1024),
            #                           nn.LeakyReLU(negative_slope=0.2))

        self.config_feature_learning_block_downsample = (
            config_feature_learning_block.downsample
        )

    def forward(self, x):
        x_list = []
        x_xyz = x.clone()
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.feature_learning_layer_list[0](x)

        if self.res_link_enable:
            res_link_list = []
            res_link_list.append(self.conv_list[0](x).max(dim=-1)[0])
            for i in range(len(self.downsample_list)):
                (x, idx_select), (_, _) = fps(
                    x, x_xyz, self.config_feature_learning_block_downsample.M[i]
                )  # (x, idx_select) = self.downsample_list[i](x, x_xyz)[0]
                x = self.feature_learning_layer_list[i + 1](x)
                x_xyz = ops.gather_by_idx(x_xyz, idx_select)
                res_link_list.append(self.conv_list[i + 1](x).max(dim=-1)[0])
            self.res_link_list = res_link_list
            x = torch.cat(res_link_list, dim=1)
            return x, res_link_list
        else:
            for i in range(len(self.downsample_list)):
                x = self.downsample_list[i](x)[0][0]
                x = self.feature_learning_layer_list[i + 1](x)
            x = self.conv(x).max(dim=-1)[0]
            return x


class FeatureLearningBlock_fps_inference_time(nn.Module):
    def __init__(self, config_feature_learning_block):
        downsample_which = config_feature_learning_block.downsample.ds_which
        ff_conv2_channels_out = (
            config_feature_learning_block.attention.ff_conv2_channels_out
        )
        self.res_link_enable = config_feature_learning_block.res_link.enable
        fl_which = config_feature_learning_block.attention.fl_which

        super(FeatureLearningBlock_fps_inference_time, self).__init__()
        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(config_feature_learning_block.embedding, layer)
                for layer in range(len(config_feature_learning_block.embedding.K))
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSample(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleWithSigma(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "global_carve":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleCarve(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local_insert":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleInsert(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "token":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleToken(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        else:
            raise NotImplementedError
        if fl_which == "n2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Neighbor2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        elif fl_which == "p2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Point2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        else:
            raise ValueError("Only n2p and p2p are valid for fl_which!")

        if self.res_link_enable:
            self.conv_list = nn.ModuleList(
                [
                    nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False)
                    for channel_in in ff_conv2_channels_out
                ]
            )
            # self.conv_list = nn.ModuleList([nn.Sequential(nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False),
            #                                           nn.BatchNorm1d(1024),
            #                                           nn.LeakyReLU(negative_slope=0.2))
            #                             for channel_in in ff_conv2_channels_out])
        else:
            self.conv = nn.Conv1d(
                ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False
            )
            # self.conv = nn.Sequential(nn.Conv1d(ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False),
            #                           nn.BatchNorm1d(1024),
            #                           nn.LeakyReLU(negative_slope=0.2))
        self.config_feature_learning_block_downsample = (
            config_feature_learning_block.downsample
        )

    def forward(self, x):
        x_list = []
        x_xyz = x.clone()
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.feature_learning_layer_list[0](x)

        inference_time_list = []
        if self.res_link_enable:
            res_link_list = []
            res_link_list.append(self.conv_list[0](x).max(dim=-1)[0])
            for i in range(len(self.downsample_list)):
                torch.cuda.synchronize()
                start_time = time.time()
                (x, idx_select), (_, _) = fps(
                    x, x_xyz, self.config_feature_learning_block_downsample.M[i]
                )  # (x, idx_select) = self.downsample_list[i](x, x_xyz)[0]
                torch.cuda.synchronize()
                inference_time_list.append(time.time() - start_time)
                x = self.feature_learning_layer_list[i + 1](x)
                x_xyz = ops.gather_by_idx(x_xyz, idx_select)
                res_link_list.append(self.conv_list[i + 1](x).max(dim=-1)[0])
            self.res_link_list = res_link_list
            x = torch.cat(res_link_list, dim=1)
            return x, res_link_list, inference_time_list
        else:
            for i in range(len(self.downsample_list)):
                x = self.downsample_list[i](x)[0][0]
                x = self.feature_learning_layer_list[i + 1](x)
            x = self.conv(x).max(dim=-1)[0]
            return x


class FeatureLearningBlock_fpsknn(nn.Module):
    def __init__(self, config_feature_learning_block):
        downsample_which = config_feature_learning_block.downsample.ds_which
        ff_conv2_channels_out = (
            config_feature_learning_block.attention.ff_conv2_channels_out
        )
        self.res_link_enable = config_feature_learning_block.res_link.enable
        fl_which = config_feature_learning_block.attention.fl_which

        super(FeatureLearningBlock_fpsknn, self).__init__()
        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(config_feature_learning_block.embedding, layer)
                for layer in range(len(config_feature_learning_block.embedding.K))
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSample(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleWithSigma(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "global_carve":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleCarve(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local_insert":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleInsert(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "token":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleToken(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        else:
            raise NotImplementedError
        if fl_which == "n2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Neighbor2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        elif fl_which == "p2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Point2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        else:
            raise ValueError("Only n2p and p2p are valid for fl_which!")

        if self.res_link_enable:
            self.conv_list = nn.ModuleList(
                [
                    nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False)
                    for channel_in in ff_conv2_channels_out
                ]
            )
            # self.conv_list = nn.ModuleList([nn.Sequential(nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False),
            #                                           nn.BatchNorm1d(1024),
            #                                           nn.LeakyReLU(negative_slope=0.2))
            #                             for channel_in in ff_conv2_channels_out])
        else:
            self.conv = nn.Conv1d(
                ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False
            )
            # self.conv = nn.Sequential(nn.Conv1d(ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False),
            #                           nn.BatchNorm1d(1024),
            #                           nn.LeakyReLU(negative_slope=0.2))

        self.config_feature_learning_block_downsample = (
            config_feature_learning_block.downsample
        )
        self.fpsknn_list = nn.ModuleList(
            [
                PointNetSetAbstraction(
                    npoint=config_feature_learning_block.downsample.M[0],
                    radius=0.2,
                    nsample=32,
                    in_channel=128,
                    mlp=[256, 256, 128],
                    group_all=False,
                ),
                PointNetSetAbstraction(
                    npoint=config_feature_learning_block.downsample.M[1],
                    radius=0.4,
                    nsample=32,
                    in_channel=128,
                    mlp=[256, 256, 128],
                    group_all=False,
                ),
            ]
        )

    def forward(self, x):
        x_list = []
        x_xyz = x.clone()
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.feature_learning_layer_list[0](x)

        if self.res_link_enable:
            res_link_list = []
            res_link_list.append(self.conv_list[0](x).max(dim=-1)[0])
            for i in range(len(self.downsample_list)):
                x, idx_select = self.fpsknn_list[i](x_xyz, x)
                # (x, idx_select), (_, _) = fps(
                #     x, x_xyz, self.config_feature_learning_block_downsample.M[i]
                # )  # (x, idx_select) = self.downsample_list[i](x, x_xyz)[0]
                x = self.feature_learning_layer_list[i + 1](x)
                x_xyz = ops.gather_by_idx(x_xyz, idx_select)
                res_link_list.append(self.conv_list[i + 1](x).max(dim=-1)[0])
            self.res_link_list = res_link_list
            x = torch.cat(res_link_list, dim=1)
            return x, res_link_list
        else:
            for i in range(len(self.downsample_list)):
                x = self.downsample_list[i](x)[0][0]
                x = self.feature_learning_layer_list[i + 1](x)
            x = self.conv(x).max(dim=-1)[0]
            return x


class FeatureLearningBlock_fpsknn_inference_time(nn.Module):
    def __init__(self, config_feature_learning_block):
        downsample_which = config_feature_learning_block.downsample.ds_which
        ff_conv2_channels_out = (
            config_feature_learning_block.attention.ff_conv2_channels_out
        )
        self.res_link_enable = config_feature_learning_block.res_link.enable
        fl_which = config_feature_learning_block.attention.fl_which

        super(FeatureLearningBlock_fpsknn_inference_time, self).__init__()
        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(config_feature_learning_block.embedding, layer)
                for layer in range(len(config_feature_learning_block.embedding.K))
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSample(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleWithSigma(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "global_carve":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleCarve(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "local_insert":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleInsert(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        elif downsample_which == "token":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleToken(
                        config_feature_learning_block.downsample, layer
                    )
                    for layer in range(len(config_feature_learning_block.downsample.M))
                ]
            )
        else:
            raise NotImplementedError
        if fl_which == "n2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Neighbor2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        elif fl_which == "p2p":
            self.feature_learning_layer_list = nn.ModuleList(
                [
                    attention.Point2PointAttention(
                        config_feature_learning_block.attention, layer
                    )
                    for layer in range(len(config_feature_learning_block.attention.K))
                ]
            )
        else:
            raise ValueError("Only n2p and p2p are valid for fl_which!")

        if self.res_link_enable:
            self.conv_list = nn.ModuleList(
                [
                    nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False)
                    for channel_in in ff_conv2_channels_out
                ]
            )
            # self.conv_list = nn.ModuleList([nn.Sequential(nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False),
            #                                           nn.BatchNorm1d(1024),
            #                                           nn.LeakyReLU(negative_slope=0.2))
            #                             for channel_in in ff_conv2_channels_out])
        else:
            self.conv = nn.Conv1d(
                ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False
            )
            # self.conv = nn.Sequential(nn.Conv1d(ff_conv2_channels_out[-1], 1024, kernel_size=1, bias=False),
            #                           nn.BatchNorm1d(1024),
            #                           nn.LeakyReLU(negative_slope=0.2))
        self.config_feature_learning_block_downsample = (
            config_feature_learning_block.downsample
        )
        self.fpsknn_list = nn.ModuleList(
            [
                PointNetSetAbstraction(
                    npoint=config_feature_learning_block.downsample.M[0],
                    radius=0.2,
                    nsample=32,
                    in_channel=128,
                    mlp=[256, 256, 128],
                    group_all=False,
                ),
                PointNetSetAbstraction(
                    npoint=config_feature_learning_block.downsample.M[1],
                    radius=0.4,
                    nsample=32,
                    in_channel=128,
                    mlp=[256, 256, 128],
                    group_all=False,
                ),
            ]
        )

    def forward(self, x):
        x_list = []
        x_xyz = x.clone()
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.feature_learning_layer_list[0](x)

        inference_time_list = []
        if self.res_link_enable:
            res_link_list = []
            res_link_list.append(self.conv_list[0](x).max(dim=-1)[0])
            for i in range(len(self.downsample_list)):
                torch.cuda.synchronize()
                start_time = time.time()
                x, idx_select = self.fpsknn_list[i](x_xyz, x)
                # (x, idx_select), (_, _) = fps(
                #     x, x_xyz, self.config_feature_learning_block_downsample.M[i]
                # )  # (x, idx_select) = self.downsample_list[i](x, x_xyz)[0]
                torch.cuda.synchronize()
                inference_time_list.append(time.time() - start_time)
                x = self.feature_learning_layer_list[i + 1](x)
                x_xyz = ops.gather_by_idx(x_xyz, idx_select)
                res_link_list.append(self.conv_list[i + 1](x).max(dim=-1)[0])
            self.res_link_list = res_link_list
            x = torch.cat(res_link_list, dim=1)
            return x, res_link_list, inference_time_list
        else:
            for i in range(len(self.downsample_list)):
                x = self.downsample_list[i](x)[0][0]
                x = self.feature_learning_layer_list[i + 1](x)
            x = self.conv(x).max(dim=-1)[0]
            return x


class Point2PointAttentionBlock(nn.Module):
    def __init__(
        self,
        egdeconv_emb_K=40,
        egdeconv_emb_group_type="center_diff",
        egdeconv_emb_conv1_in=6,
        egdeconv_emb_conv1_out=64,
        egdeconv_emb_conv2_in=64,
        egdeconv_emb_conv2_out=64,
        downsample_which="p2p",
        downsample_M=(1024, 512),
        downsample_q_in=(64, 64),
        downsample_q_out=(64, 64),
        downsample_k_in=(64, 64),
        downsample_k_out=(64, 64),
        downsample_v_in=(64, 64),
        downsample_v_out=(64, 64),
        downsample_num_heads=(1, 1),
        q_in=(64, 64, 64),
        q_out=(64, 64, 64),
        k_in=(64, 64, 64),
        k_out=(64, 64, 64),
        v_in=(64, 64, 64),
        v_out=(64, 64, 64),
        num_heads=(8, 8, 8),
        ff_conv1_channels_in=(64, 64, 64),
        ff_conv1_channels_out=(128, 128, 128),
        ff_conv2_channels_in=(128, 128, 128),
        ff_conv2_channels_out=(64, 64, 64),
    ):
        super(Point2PointAttentionBlock, self).__init__()
        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(
                    emb_k,
                    emb_g_type,
                    emb_conv1_in,
                    emb_conv1_out,
                    emb_conv2_in,
                    emb_conv2_out,
                )
                for emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in zip(
                    egdeconv_emb_K,
                    egdeconv_emb_group_type,
                    egdeconv_emb_conv1_in,
                    egdeconv_emb_conv1_out,
                    egdeconv_emb_conv2_in,
                    egdeconv_emb_conv2_out,
                )
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSample(
                        ds_M,
                        ds_q_in,
                        ds_q_out,
                        ds_k_in,
                        ds_k_out,
                        ds_v_in,
                        ds_v_out,
                        ds_heads,
                    )
                    for ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(
                        downsample_M,
                        downsample_q_in,
                        downsample_q_out,
                        downsample_k_in,
                        downsample_k_out,
                        downsample_v_in,
                        downsample_v_out,
                        downsample_num_heads,
                    )
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleWithSigma(
                        ds_M,
                        ds_q_in,
                        ds_q_out,
                        ds_k_in,
                        ds_k_out,
                        ds_v_in,
                        ds_v_out,
                        ds_heads,
                    )
                    for ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(
                        downsample_M,
                        downsample_q_in,
                        downsample_q_out,
                        downsample_k_in,
                        downsample_k_out,
                        downsample_v_in,
                        downsample_v_out,
                        downsample_num_heads,
                    )
                ]
            )
        else:
            raise ValueError("Only global and local are valid for ds_which!")
        self.point2point_list = nn.ModuleList(
            [
                attention.Point2PointAttention(
                    q_input,
                    q_output,
                    k_input,
                    k_output,
                    v_input,
                    v_output,
                    heads,
                    ff_conv1_channel_in,
                    ff_conv1_channel_out,
                    ff_conv2_channel_in,
                    ff_conv2_channel_out,
                )
                for q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out in zip(
                    q_in,
                    q_out,
                    k_in,
                    k_out,
                    v_in,
                    v_out,
                    num_heads,
                    ff_conv1_channels_in,
                    ff_conv1_channels_out,
                    ff_conv2_channels_in,
                    ff_conv2_channels_out,
                )
            ]
        )
        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False)
                for channel_in in ff_conv2_channels_out
            ]
        )

    def forward(self, x):
        x_list = []
        res_link_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.point2point_list[0](x)
        res_link_list.append(self.conv_list[0](x).max(dim=-1)[0])
        for i in range(len(self.downsample_list)):
            x = self.downsample_list[i](x)[0][0]
            x = self.point2point_list[i + 1](x)
            res_link_list.append(self.conv_list[i + 1](x).max(dim=-1)[0])
        x = torch.cat(res_link_list, dim=1)
        return x


class EdgeConvBlock(nn.Module):
    def __init__(
        self,
        egdeconv_emb_K=40,
        egdeconv_emb_group_type="center_diff",
        egdeconv_emb_conv1_in=6,
        egdeconv_emb_conv1_out=64,
        egdeconv_emb_conv2_in=64,
        egdeconv_emb_conv2_out=64,
        downsample_which="p2p",
        downsample_M=(1024, 512),
        downsample_q_in=(64, 64),
        downsample_q_out=(64, 64),
        downsample_k_in=(64, 64),
        downsample_k_out=(64, 64),
        downsample_v_in=(64, 64),
        downsample_v_out=(64, 64),
        downsample_num_heads=(1, 1),
        K=(32, 32, 32),
        group_type=("center_diff", "center_diff", "center_diff"),
        conv1_channel_in=(3 * 2, 64 * 2, 64 * 2),
        conv1_channel_out=(64, 64, 64),
        conv2_channel_in=(64, 64, 64),
        conv2_channel_out=(64, 64, 64),
    ):
        super(EdgeConvBlock, self).__init__()
        self.embedding_list = nn.ModuleList(
            [
                embedding.EdgeConv(
                    emb_k,
                    emb_g_type,
                    emb_conv1_in,
                    emb_conv1_out,
                    emb_conv2_in,
                    emb_conv2_out,
                )
                for emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in zip(
                    egdeconv_emb_K,
                    egdeconv_emb_group_type,
                    egdeconv_emb_conv1_in,
                    egdeconv_emb_conv1_out,
                    egdeconv_emb_conv2_in,
                    egdeconv_emb_conv2_out,
                )
            ]
        )
        if downsample_which == "global":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSample(
                        ds_M,
                        ds_q_in,
                        ds_q_out,
                        ds_k_in,
                        ds_k_out,
                        ds_v_in,
                        ds_v_out,
                        ds_heads,
                    )
                    for ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(
                        downsample_M,
                        downsample_q_in,
                        downsample_q_out,
                        downsample_k_in,
                        downsample_k_out,
                        downsample_v_in,
                        downsample_v_out,
                        downsample_num_heads,
                    )
                ]
            )
        elif downsample_which == "local":
            self.downsample_list = nn.ModuleList(
                [
                    downsample.DownSampleWithSigma(
                        ds_M,
                        ds_q_in,
                        ds_q_out,
                        ds_k_in,
                        ds_k_out,
                        ds_v_in,
                        ds_v_out,
                        ds_heads,
                    )
                    for ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(
                        downsample_M,
                        downsample_q_in,
                        downsample_q_out,
                        downsample_k_in,
                        downsample_k_out,
                        downsample_v_in,
                        downsample_v_out,
                        downsample_num_heads,
                    )
                ]
            )
        else:
            raise ValueError("Only global and local are valid for ds_which!")
        self.edgeconv_list = nn.ModuleList(
            [
                embedding.EdgeConv(k, g_type, conv1_in, conv1_out, conv2_in, conv2_out)
                for k, g_type, conv1_in, conv1_out, conv2_in, conv2_out in zip(
                    K,
                    group_type,
                    conv1_channel_in,
                    conv1_channel_out,
                    conv2_channel_in,
                    conv2_channel_out,
                )
            ]
        )
        # self.conv_list = nn.ModuleList([nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False) for channel_in in conv2_channel_out])
        self.conv_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channel_in, 1024, kernel_size=1, bias=False),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(negative_slope=0.2),
                )
                for channel_in in conv2_channel_out
            ]
        )

    def forward(self, x):
        x_list = []
        res_link_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.edgeconv_list[0](x)
        res_link_list.append(self.conv_list[0](x).max(dim=-1)[0])
        for i in range(len(self.downsample_list)):
            x = self.downsample_list[i](x)[0][0]
            x = self.edgeconv_list[i + 1](x)
            res_link_list.append(self.conv_list[i + 1](x).max(dim=-1)[0])
        x = torch.cat(res_link_list, dim=1)
        return x
