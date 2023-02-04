import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np


def knn(x, k):
    """
    Input:
        points: input points data_process, [B, N, C]
    Return:
        idx: sample index data_process, [B, N, K]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def index_points_neighbors(x, idx):
    """
    Input:
        points: input points data_process, [B, N, C]
        idx: sample index data_process, [B, N, K]
    Return:
        new_points:, indexed points data_process, [B, N, K, C]
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
        points: input points data_process, [B, C, N]
    Return:
        feature_points:, indexed points data_process, [B, 2*C, N, K]
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
    """
        self-attention model
    """

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


def create_conv1d_serials(channel_list):
    conv1d_serials = nn.ModuleList(
        [
            nn.Conv1d(
                in_channels=channel_list[i],
                out_channels=channel_list[i + 1],
                kernel_size=1,
            )
            for i in range(len(channel_list) - 1)
        ]
    )

    return conv1d_serials


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)
    elif type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):  # [bs,4,4096,16]
        attn = q @ k.transpose(-1, -2)  # [bs,4,4096,4096]
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-1)
        output = attn @ v  # [bs,4,4096,16]

        return output, attn


class SA_Layer_Multi_Head(nn.Module):
    def __init__(self, args, num_features):  # input [bs,n_points,num_features]
        super(SA_Layer_Multi_Head, self).__init__()
        self.num_heads = args.num_heads
        self.num_hidden_features = args.self_encoder_latent_features
        self.num_features = num_features

        self.w_qs = nn.Linear(self.num_features, self.num_heads * int(self.num_hidden_features / self.num_heads),
                              bias=False)
        self.w_ks = nn.Linear(self.num_features, self.num_heads * int(self.num_hidden_features / self.num_heads),
                              bias=False)
        self.w_vs = nn.Linear(self.num_features, self.num_heads * int(self.num_hidden_features / self.num_heads),
                              bias=False)
        self.attention = Attention()
        self.norm1 = nn.LayerNorm(self.num_hidden_features)
        self.trans = nn.Linear(self.num_hidden_features, self.num_features)
        self.norm2 = nn.LayerNorm(self.num_features)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b_s, n_points, _ = x.size()
        original = x
        q = self.w_qs(x).view(-1, n_points, self.num_heads,
                              int(self.num_hidden_features / self.num_heads))  # [bs,4096,4,32]
        k = self.w_ks(x).view(-1, n_points, self.num_heads,
                              int(self.num_hidden_features / self.num_heads))  # [bs,4096,4,32]
        v = self.w_vs(x).view(-1, n_points, self.num_heads,
                              int(self.num_hidden_features / self.num_heads))  # [bs,4096,4,32]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,
                                                                    2)  # [bs,4,4096,32] [bs,4,4096,32]  [bs,4,4096,32]
        q, attn = self.attention(q, k, v)  # [bs,4,4096,32]
        q = q.transpose(1, 2).contiguous().view(b_s, n_points, -1)  # [bs,4096,128]
        q = self.norm1(q)
        ######################
        # x=self.norm2(self.trans(q)+original)
        ###########################
        residual = self.act(self.norm2(original - self.trans(q)))
        x = original + residual
        return x


def create_conv3d_serials(channel_list, num_points, dim):
    conv3d_serials = nn.ModuleList(
        [
            nn.Conv3d(
                in_channels=channel_list[i],
                out_channels=channel_list[i + 1],
                kernel_size=(1, 1, 1),
            )
            for i in range(len(channel_list) - 1)
        ]
    )
    conv3d_serials.insert(
        0,
        nn.Conv3d(
            in_channels=1,
            out_channels=channel_list[0],
            kernel_size=(1, num_points, dim),
        ),
    )

    return conv3d_serials


class SA_Layers(nn.Module):
    def __init__(self, n_layers, encoder_layer):
        super(SA_Layers, self).__init__()
        self.num_layers = n_layers
        self.encoder_layer = encoder_layer
        self.layers = nn.ModuleList([copy.deepcopy(self.encoder_layer) for i in range(self.num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


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


class PTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, last_dim=256, dropout=0.1, activation=F.relu):
        super(PTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, 512)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, last_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        return tgt


class PTransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, num_layers, last_layer, norm=None):
        super(PTransformerDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for i in range(num_layers)])  # repeat the decoder layers
        self.last_layer = last_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm:
            output = self.norm(output)

        output = self.last_layer(output, memory)

        return output


class PCT_patch_semseg(nn.Module):
    def __init__(self, args):
        super(PCT_patch_semseg, self).__init__()
        self.source_sample_after_rotate = args.after_stn_as_kernel_neighbor_query
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 32
        self.d_model = 512
        self.args = args
        self.k = args.k
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
        self.conv5 = nn.Conv1d(1024 * 2, 512, 1)
        self.dp5 = nn.Dropout(0.5)

        #############################################################################
        # self desigened superpoint sample net and its net of generation local features
        #############################################################################
        self.sort_ch = [self.input_dim, 64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i + 1])
                for i in range(len(self.sort_ch) - 1)
            ]
        )
        self.superpointnet = ball_query_sample_with_goal(args, self.sort_ch[-1], self.input_dim, self.actv_fn,
                                                         top_k=self.top_k)

        #############################
        #
        #############################
        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=512)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 1, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model, nhead=4, num_encoder_layers=1,
                                                num_decoder_layers=1, custom_decoder=self.custom_decoder, )
        # self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=1,num_decoder_layers=1)
        self.transformer_model.apply(init_weights)

        ##########################################################
        # final segmentation layer
        ######################################################
        self.bnup = nn.BatchNorm1d(1024)
        self.convup = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                    self.bnup,  # 2048
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.conv7 = nn.Conv1d(256, 6, 1)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp6 = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x, target):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.float()
        input = x

        trans = self.s3n(x)
        x = x.permute(0, 2, 1)  # (batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        # Visuell_PointCloud_per_batch(x,target)
        x = x.permute(0, 2, 1)
        x_a_r = x

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
        # global info
        x_global = x

        x = self.conv__(x)  # (batch_size, 512, num_points)->(batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = x.unsqueeze(-1).repeat(1, 1, num_points)  # (batch_size, 1024)->(batch_size, 1024, num_points)
        # x12=torch.mean(x,dim=2,keepdim=False)       # (batch_size, 1024, num_points) -> (batch_size,1024)
        # x12=x12.unsqueeze(-1).repeat(1,1,num_points)# (batch_size, 1024)->(batch_size, 1024, num_points)
        # x_global_integrated = torch.cat((x11, x12), dim=1)     # (batch_size,1024,num_points)+(batch_size, 1024,num_points)-> (batch_size, 2048,num_points)
        x = torch.cat((x, x_global),
                      dim=1)  # (batch_size,2048,num_points)+(batch_size, 1024,num_points) ->(batch_size, 3036,num_points)

        #############################################
        ## sample Features
        #############################################
        if self.source_sample_after_rotate:
            x_sample = x_a_r
        else:
            x_sample = input

        for i, sort_conv in enumerate(self.sort_cnn):  # [bs,1,Cor,n_points]->[bs,256,n_points]

            bn = self.sort_bn[i]
            x_sample = self.actv_fn(bn(sort_conv(x_sample)))

        # patch
        x_patch, result = self.superpointnet(x_sample, input, x_a_r, target)

        #############################################
        ## Point Transformer
        #############################################
        target = x_global.permute(2, 0, 1)  # [bs,1024,64]->[64,bs,1024]
        source = x_patch.permute(2, 0, 1)  # [bs,1024,10]->[10,bs,1024]
        embedding = self.transformer_model(source, target)  # [64,bs,1024]+[16,bs,1024]->[16,bs,1024]
        embedding = embedding.permute(1, 2, 0)
        # embedding=self.convup(embedding)

        ################################################
        ##segmentation
        #################################################
        # embedding = embedding.max(dim=-1, keepdim=False)[0]
        # embedding=embedding.unsqueeze(-1).repeat(1,1,num_points)
        x = torch.cat((x, embedding), dim=1)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(embedding)))  # (batch_size, 512,num_points) ->(batch_size,256,num_points)
        x = self.dp6(x)
        x = self.conv7(x)  # # (batch_size, 256,num_points) ->(batch_size,6,num_points)

        if self.args.training:
            return x, trans, result
        else:
            return x, trans, None


if __name__ == '__main__':
    pass
