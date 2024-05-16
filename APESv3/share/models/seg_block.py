import torch
from torch import nn
from utils import ops
from models import embedding
from models import attention
from models import downsample
from models import upsample

class Neighbor2PointAttentionBlock(nn.Module):
    def __init__(self, config_n2p_block):
        super(Neighbor2PointAttentionBlock, self).__init__()
        downsample_which = config_n2p_block.downsample.ds_which
        upsample_which   = config_n2p_block.upsample.us_which

        self.embedding_list = nn.ModuleList([embedding.EdgeConv(config_n2p_block.embedding, layer) for layer in range(len(config_n2p_block.embedding.K))])
        if downsample_which == 'global':
            self.downsample_list = nn.ModuleList([downsample.DownSample(config_n2p_block.downsample, layer) for layer in range(len(config_n2p_block.downsample.M))])
        elif downsample_which == 'local':
            self.downsample_list = nn.ModuleList([downsample.DownSampleWithSigma(config_n2p_block.downsample, layer) for layer in range(len(config_n2p_block.downsample.M))])
        elif downsample_which == 'global_carve':
            self.downsample_list = nn.ModuleList([downsample.DownSampleCarve(config_n2p_block.downsample, layer) for layer in range(len(config_n2p_block.downsample.M))])
        elif downsample_which == 'local_insert':
            self.downsample_list = nn.ModuleList([downsample.DownSampleInsert(config_n2p_block.downsample, layer) for layer in range(len(config_n2p_block.downsample.M))])
        else:
            raise ValueError('Only global_carve and local_insert are valid for ds_which!')
        self.neighbor2point_list = nn.ModuleList([attention.Neighbor2PointAttention(config_n2p_block.attention, layer) for layer in range(len(config_n2p_block.attention.K))])
        
        if  upsample_which  == 'crossA':
            self.upsample_list = nn.ModuleList([upsample.UpSample(config_n2p_block.upsample, layer) for layer in range(len(config_n2p_block.upsample.q_in))])
        elif upsample_which == 'selfA':
            self.upsample_list = nn.ModuleList([upsample.UpSampleSelfAttention(config_n2p_block.upsample, layer) for layer in range(len(config_n2p_block.upsample.q_in))])
        elif upsample_which == 'interpolation':
            self.upsample_list = nn.ModuleList([upsample.UpSampleInterpolation(config_n2p_block.upsample, layer) for layer in range(len(config_n2p_block.upsample.q_in))])
        else:
            raise ValueError('Only crossA and selfA are valid for us_which!')

    def forward(self, x):
        x_xyz = x[:, :3, :]
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.neighbor2point_list[0](x)
        x_list = [x]
        points_drop_list = []
        idx_select_list = []
        idx_drop_list = []
        x_xyz_list = [x_xyz]
        for i in range(len(self.downsample_list)):
            (x, idx_select), (points_drop, idx_drop) = self.downsample_list[i](x, x_xyz)
            x = self.neighbor2point_list[i+1](x)
            x_xyz = ops.gather_by_idx(x_xyz, idx_select)
            x_list.append(x)
            x_xyz_list.append(x_xyz)
            points_drop_list.append(points_drop)
            idx_select_list.append(idx_select)
            idx_drop_list.append(idx_drop)
        split = int((len(self.neighbor2point_list) - 1) / 2)
        x = ((x_list.pop(), idx_select_list.pop(), x_xyz_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        for j in range(len(self.upsample_list)):
            x_tmp = x_list.pop()
            x_xyz_tmp = x_xyz_list[-1-j]
            x = self.upsample_list[j](x_tmp, x, x_xyz_tmp)
            x = self.neighbor2point_list[j+1+split](x)
            if j < len(self.upsample_list) - 1:
                x = ((x, idx_select_list.pop(), x_xyz_list[-1-j]), (points_drop_list.pop(), idx_drop_list.pop()))
        return x

class Point2PointAttentionBlock(nn.Module):
    def __init__(self, egdeconv_emb_K=40, egdeconv_emb_group_type='center_diff',
                 egdeconv_emb_conv1_in=6, egdeconv_emb_conv1_out=64, egdeconv_emb_conv2_in=64, egdeconv_emb_conv2_out=64,
                 downsample_which='p2p', downsample_M=(1024, 512), downsample_q_in=(64, 64), downsample_q_out=(64, 64),
                 downsample_k_in=(64, 64), downsample_k_out=(64, 64), downsample_v_in=(64, 64),
                 downsample_v_out=(64, 64), downsample_num_heads=(1, 1),
                 upsample_which='crossA', upsample_q_in=(64, 64), upsample_q_out=(64, 64),
                 upsample_k_in=(64, 64), upsample_k_out=(64, 64), upsample_v_in=(64, 64),
                 upsample_v_out=(64, 64), upsample_num_heads=(1, 1),
                 q_in=(64, 64, 64), q_out=(64, 64, 64), k_in=(64, 64, 64), k_out=(64, 64, 64), v_in=(64, 64, 64), v_out=(64, 64, 64), num_heads=(8, 8, 8),
                 ff_conv1_channels_in=(64, 64, 64), ff_conv1_channels_out=(128, 128, 128),
                 ff_conv2_channels_in=(128, 128, 128), ff_conv2_channels_out=(64, 64, 64)):
        super(Point2PointAttentionBlock, self).__init__()
        self.embedding_list = nn.ModuleList([embedding.EdgeConv(emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out) for emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in zip(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in, egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out)])
        if downsample_which == 'global':
            self.downsample_list = nn.ModuleList([downsample.DownSample(ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_M,  downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        elif downsample_which == 'local':
            self.downsample_list = nn.ModuleList([downsample.DownSampleWithSigma(ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_M,  downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        else:
            raise ValueError('Only global and local are valid for ds_which!')
        if upsample_which == 'crossA':
            self.upsample_list = nn.ModuleList([upsample.UpSample(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out, upsample_num_heads)])
        elif upsample_which == 'selfA':
            self.upsample_list = nn.ModuleList([upsample.UpSampleSelfAttention(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out, upsample_num_heads)])
        else:
            raise ValueError('Only crossA and selfA are valid for us_which!')
        self.point2point_list = nn.ModuleList([attention.Point2PointAttention(q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out)
                                 for q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out
                                 in zip(q_in, q_out, k_in, k_out, v_in, v_out, num_heads, ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out)])

    def forward(self, x):
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.point2point_list[0](x)
        x_list = [x]
        points_drop_list = []
        idx_select_list = []
        idx_drop_list = []
        for i in range(len(self.downsample_list)):
            (x, idx_select), (points_drop, idx_drop) = self.downsample_list[i](x)
            x = self.point2point_list[i+1](x)
            x_list.append(x)
            points_drop_list.append(points_drop)
            idx_select_list.append(idx_select)
            idx_drop_list.append(idx_drop)
        split = int((len(self.point2point_list) - 1) / 2)
        x = ((x_list.pop(), idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        for j in range(len(self.upsample_list)):
            x_tmp = x_list.pop()
            x = self.upsample_list[j](x_tmp, x)
            x = self.point2point_list[j+1+split](x)
            if j < len(self.upsample_list) - 1:
                x = ((x, idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        return x

class EdgeConvBlock(nn.Module):
    def __init__(self, egdeconv_emb_K=40, egdeconv_emb_group_type='center_diff',
                 egdeconv_emb_conv1_in=6, egdeconv_emb_conv1_out=64, egdeconv_emb_conv2_in=64, egdeconv_emb_conv2_out=64,
                 downsample_which='p2p', downsample_M=(1024, 512), downsample_q_in=(64, 64), downsample_q_out=(64, 64),
                 downsample_k_in=(64, 64), downsample_k_out=(64, 64), downsample_v_in=(64, 64),
                 downsample_v_out=(64, 64), downsample_num_heads=(1, 1),
                 upsample_which='crossA', upsample_q_in=(64, 64), upsample_q_out=(64, 64),
                 upsample_k_in=(64, 64), upsample_k_out=(64, 64), upsample_v_in=(64, 64),
                 upsample_v_out=(64, 64), upsample_num_heads=(1, 1),
                 K=(32, 32, 32), group_type=('center_diff', 'center_diff', 'center_diff'),
                 conv1_channel_in=(3 * 2, 64 * 2, 64 * 2), conv1_channel_out=(64, 64, 64),
                 conv2_channel_in=(64, 64, 64), conv2_channel_out=(64, 64, 64)):
        super(EdgeConvBlock, self).__init__()
        self.embedding_list = nn.ModuleList([embedding.EdgeConv(emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out) for emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in zip(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in, egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out)])
        if downsample_which == 'global':
            self.downsample_list = nn.ModuleList([downsample.DownSample(ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_M,  downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        elif downsample_which == 'local':
            self.downsample_list = nn.ModuleList([downsample.DownSampleWithSigma(ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_M, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_M,  downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        else:
            raise ValueError('Only global and local are valid for ds_which!')
        if upsample_which == 'crossA':
            self.upsample_list = nn.ModuleList([upsample.UpSample(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out, upsample_num_heads)])
        elif upsample_which == 'selfA':
            self.upsample_list = nn.ModuleList([upsample.UpSampleSelfAttention(us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads) for us_q_in, us_q_out, us_k_in, us_k_out, us_v_in, us_v_out, us_heads in zip(upsample_q_in, upsample_q_out, upsample_k_in, upsample_k_out, upsample_v_in, upsample_v_out, upsample_num_heads)])
        else:
            raise ValueError('Only crossA and selfA are valid for us_which!')
        self.edgeconv_list = nn.ModuleList([embedding.EdgeConv(k, g_type, conv1_in, conv1_out, conv2_in, conv2_out) for k, g_type, conv1_in, conv1_out, conv2_in, conv2_out in zip(K, group_type, conv1_channel_in, conv1_channel_out, conv2_channel_in, conv2_channel_out)])

    def forward(self, x):
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.edgeconv_list[0](x)
        x_list = [x]
        points_drop_list = []
        idx_select_list = []
        idx_drop_list = []
        for i in range(len(self.downsample_list)):
            (x, idx_select), (points_drop, idx_drop) = self.downsample_list[i](x)
            x = self.edgeconv_list[i+1](x)
            x_list.append(x)
            points_drop_list.append(points_drop)
            idx_select_list.append(idx_select)
            idx_drop_list.append(idx_drop)
        split = int((len(self.edgeconv_list) - 1) / 2)
        x = ((x_list.pop(), idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        for j in range(len(self.upsample_list)):
            x_tmp = x_list.pop()
            x = self.upsample_list[j](x_tmp, x)
            x = self.edgeconv_list[j+1+split](x)
            if j < len(self.upsample_list)-1:
                x = ((x, idx_select_list.pop()), (points_drop_list.pop(), idx_drop_list.pop()))
        return x
