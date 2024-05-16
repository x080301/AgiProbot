import torch
from torch import nn
from models import comp_block
import torch.nn.functional as F
from utils.pointnet_utils import PointNetEncoder

class PointNet(nn.Module):
    def __init__(self, k=40, normal_channel=False):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x  # x.shape == (B, 40)

class ComparisonModel(nn.Module):
    def __init__(self, config):

        super(ComparisonModel, self).__init__()

        if config.neighbor2point_block.enable:
            self.block = comp_block.Neighbor2PointAttentionBlock(config.neighbor2point_block)
            nfeat = config.neighbor2point_block.attention.ff_conv2_channels_out[-1]
        else:
            raise ValueError('This time only support neighbor2point block!')
        self.conv = nn.Conv1d(nfeat, 3, 1, bias=False)
        self.target_model = PointNet()
    
    def forward(self, x): # x.shape == (B, 3, N)
        x = self.block(x)           # x.shape == (B, C, M)
        x = self.conv(x)            # x.shape == (B, 3, M)
        x = self.target_model(x)    # x.shape == (B, 40)
        return x