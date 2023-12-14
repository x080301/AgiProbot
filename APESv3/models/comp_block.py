import torch
from torch import nn
from utils import ops
from models import embedding
from models import attention
from models import downsample


class Neighbor2PointAttentionBlock(nn.Module):
    def __init__(self, config_n2p_block):
        downsample_which        = config_n2p_block.downsample.ds_which
        super(Neighbor2PointAttentionBlock, self).__init__()
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
    def forward(self, x):
        x_list = []
        x_xyz = x.clone()
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.neighbor2point_list[0](x)
        for i in range(len(self.downsample_list)):
            (x, idx_select) = self.downsample_list[i](x, x_xyz)[0]
            x = self.neighbor2point_list[i+1](x)
            x_xyz = ops.gather_by_idx(x_xyz, idx_select)
        return x