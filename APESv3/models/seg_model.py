import torch
from torch import nn
from models import seg_block
from models import embedding
from utils import ops

class ShapeNetModel(nn.Module):
    def __init__(self, config):

        super(ShapeNetModel, self).__init__()

        # num_enabled_blocks = neighbor2point_enable + point2point_enable + edgeconv_enable
        # if num_enabled_blocks != 1:
        #     raise ValueError(f'Only one of neighbor2point_block, point2point_block and edgecov_block should be enabled, but got {num_enabled_blocks} block(s) enabled!')
        if config.neighbor2point_block.enable:
            self.block = seg_block.Neighbor2PointAttentionBlock(config.neighbor2point_block)
            output_channels = config.neighbor2point_block.attention.ff_conv2_channels_out[-1]
        # if point2point_enable:
        #     self.block = seg_block.Point2PointAttentionBlock(point2point_egdeconv_emb_K, point2point_egdeconv_emb_group_type,
        #                                            point2point_egdeconv_emb_conv1_in, point2point_egdeconv_emb_conv1_out, point2point_egdeconv_emb_conv2_in, point2point_egdeconv_emb_conv2_out,
        #                                            point2point_down_which, point2point_downsample_M, point2point_down_q_in, point2point_down_q_out, point2point_down_k_in,
        #                                            point2point_down_k_out, point2point_down_v_in, point2point_down_v_out, point2point_down_num_heads, point2point_up_which, point2point_up_q_in, point2point_up_q_out, point2point_up_k_in,
        #                                            point2point_up_k_out, point2point_up_v_in, point2point_up_v_out, point2point_up_num_heads,
        #                                            point2point_q_in, point2point_q_out, point2point_k_in, point2point_k_out, point2point_v_in,
        #                                            point2point_v_out, point2point_num_heads, point2point_ff_conv1_in, point2point_ff_conv1_out, point2point_ff_conv2_in, point2point_ff_conv2_out)
        #     output_channels = point2point_ff_conv2_out[-1]
        # if edgeconv_enable:
        #     self.block = seg_block.EdgeConvBlock(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in,
        #                                egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out,
        #                                edgeconv_downsample_which, edgeconv_downsample_M, edgeconv_downsample_q_in, edgeconv_downsample_q_out,
        #                                edgeconv_downsample_k_in, edgeconv_downsample_k_out, edgeconv_downsample_v_in,
        #                                edgeconv_downsample_v_out, edgeconv_downsample_num_heads,
        #                                edgeconv_upsample_which, edgeconv_upsample_q_in, edgeconv_upsample_q_out,
        #                                edgeconv_upsample_k_in, edgeconv_upsample_k_out, edgeconv_upsample_v_in,
        #                                edgeconv_upsample_v_out, edgeconv_upsample_num_heads,
        #                                edgeconv_K, edgeconv_group_type, edgeconv_conv1_channel_in, edgeconv_conv1_channel_out,
        #                                edgeconv_conv2_channel_in, edgeconv_conv2_channel_out)
        #     output_channels = edgeconv_conv2_channel_out[-1]

        self.conv = nn.Sequential(nn.Conv1d(output_channels, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(output_channels+2048+64, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(1024, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv4 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(128),
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(256, 50, kernel_size=1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.STN_enable = config.neighbor2point_block.STN
        if self.STN_enable == True:
            self.STN = embedding.STN()

    def forward(self, x, category_id):
        # x.shape == (B, 3, N)  category_id.shape == (B, 16, 1)
        B, C, N = x.shape
        # x.shape == (B, 3, N)
        
        if self.STN_enable == True:
            x0, _ = ops.group(x, 32, 'center_diff')  # (B, 3, num_points) -> (B, 3*2, num_points, k)
            t = self.STN(x0)  # (B, 3, 3)
            x = x.transpose(2, 1)  # (B, 3, num_points) -> (B, num_points, 3)
            x = torch.bmm(x, t)  # (B, num_points, 3) * (B, 3, 3) -> (B, num_points, 3)
            x = x.transpose(2, 1)  # (B, num_points, 3) -> (B, 3, num_points)
        
        x_tmp = self.block(x)
        # x_tmp.shape == (B, C, N)
        x = self.conv(x_tmp)
        # x.shape == (B, 1024, N)
        x_max = x.max(dim=-1, keepdim=True)[0]
        # x_max.shape == (B, 1024, 1)
        x_average = x.mean(dim=-1, keepdim=True)
        # x_average.shape == (B, 1024, 1)
        x = torch.cat([x_max, x_average], dim=1)
        # x.shape == (B, 2048, 1)
        category_id = self.conv1(category_id)
        # category_id.shape == (B, 64, 1)
        x = torch.cat([x, category_id], dim=1)
        # x.shape === (B, 2048+64, 1)
        x = x.repeat(1, 1, N)
        # x.shape == (B, 2048+64, N)
        x = torch.cat([x, x_tmp], dim=1)
        # x.shape == (B, 2048+64+C, N)
        x = self.conv2(x)
        # x.shape == (B, 1024, N)
        x = self.dp1(x)
        # x.shape == (B, 1024, N)
        x = self.conv3(x)
        # x.shape == (B, 256, N)
        x = self.dp2(x)
        # x.shape == (B, 256, N)
        x = self.conv4(x)
        # x.shape == (B, 50, N)
        # x = self.conv5(x)
        # # x.shape == (B, 50, N)
        return x