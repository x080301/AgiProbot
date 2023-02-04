import torch
from torch import nn
import torch.nn.functional as F


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


class SALayerSingleHead(nn.Module):
    def __init__(self, channels):
        super(SALayerSingleHead, self).__init__()
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
        x_r = x @ attention  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        x = x.permute(0, 2, 1)
        return x


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(channel, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),

            nn.ReLU()
        )
        self.fc = nn.Linear(256, 9)

    def forward(self, x):
        batchsize = x.size()[0]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc(x)

        iden = torch.autograd.Variable(torch.eye(3, dtype=torch.float32)).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden

        x = x.view(-1, 3, 3)

        return x


class PCT_semseg(nn.Module):
    def __init__(self, args):
        super(PCT_semseg, self).__init__()
        self.args = args
        self.k = args.k  # k nearest neighbors

        self.s3n = STN3d(3)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.sa1 = SALayerSingleHead(128)
        self.sa2 = SALayerSingleHead(128)
        self.sa3 = SALayerSingleHead(128)
        self.sa4 = SALayerSingleHead(128)
        self.bnmax11 = nn.BatchNorm1d(64)
        self.bnmax12 = nn.BatchNorm1d(64)

        self.lbr1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),  # 3*64=384
                                  self.bn1,  # 2*64*2=256
                                  nn.LeakyReLU(negative_slope=0.2))  # 0
        self.lbr2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),  # 64*64=4096
                                  self.bn2,  # 256
                                  nn.LeakyReLU(negative_slope=0.2))  # 0
        self.lbr3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),  # 128*64=8096
                                  self.bn3,  # 256
                                  nn.LeakyReLU(negative_slope=0.2))  # 0
        self.lbr4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),  # 64*64=4096
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

        # ************************* #
        # Input Embedding
        # ************************* #
        # spatial transform
        trans = self.s3n(x)
        x = x.permute(0, 2, 1)  # (batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        # Visuell_PointCloud_per_batch(x,target)
        x = x.permute(0, 2, 1)

        # EdgeConv
        x = get_neighbors(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.lbr1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.lbr2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # EdgeConv
        x = get_neighbors(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.lbr3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.lbr4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2), dim=1)  # (batch_size, 64, num_points, k)*2 ->(batch_size, 128, num_points)

        # ************************* #
        # 4 stacked Attention module
        # ************************* #
        x = x.permute(0, 2, 1)
        x1 = self.sa1(x)  # (batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)  50MB
        x2 = self.sa2(x1)  # (batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x3 = self.sa3(x2)  # (batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x4 = self.sa4(x3)  # (batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # (batch_size, 64*2, num_points)*4->(batch_size, 512, num_points)

        # ************************* #
        # point and global feature
        # ************************* #
        # point feature
        x = x.permute(0, 2, 1)
        x__ = x
        x = self.conv__(x)  # (batch_size, 512, num_points)->(batch_size, 1024, num_points)
        point_feture = x

        # global feature
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = x.unsqueeze(-1).repeat(1, 1, num_points)  # (batch_size, 1024)->(batch_size, 1024, num_points)

        x = torch.cat((x, x__),
                      dim=1)  # (batch_size,2048,num_points)+(batch_size, 1024,num_points) ->(batch_size, 3036,num_points)

        # ************************* #
        # Segmentation
        # ************************* #
        # LBRD
        x = self.relu(self.bn5(self.conv5(x)))  # (batch_size, 3036,num_points)-> (batch_size, 512,num_points)
        x = self.dp5(x)

        # LBR
        x = self.relu(self.bn6(self.conv6(x)))  # (batch_size, 512,num_points) ->(batch_size,256,num_points)

        # Linear
        x = self.conv7(x)  # # (batch_size, 256,num_points) ->(batch_size,6,num_points)

        # ************************* #
        # Classification
        # ************************* #
        y1 = F.adaptive_max_pool1d(point_feture, 1).view(batch_size,
                                                         -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        y2 = F.adaptive_avg_pool1d(point_feture, 1).view(batch_size,
                                                         -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        y = torch.cat((y1, y2), 1)  # (batch_size, emb_dims*2)

        # LBRD
        y = F.leaky_relu(self.bn9(self.linear1(y)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        y = self.dp2(y)

        # LBRD
        y = F.leaky_relu(self.bn10(self.linear2(y)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        y = self.dp3(y)

        # Linear
        y = self.linear3(y)  # (batch_size, 256) -> (batch_size, 5)
        return x, trans, y, None
