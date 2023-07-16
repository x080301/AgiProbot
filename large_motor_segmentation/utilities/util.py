import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_token_loss(bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals, ground_truth, args):
    # (B,2,T), (B,bolt_type,T), (B,3,T), (B,3,T), (B,2+bolt_type+3+3,T)

    loss = 0

    if not args.model.token.bolt_existing_loss == 0:
        logits = bolt_existing_label.permute(0, 2, 1)
        logits = logits.view(-1, args.num_segmentation_type).contiguous()
        # _                                                 (B,2,T) -> (B*T,2)
        target = ground_truth[:, 0:2].view(-1)
        target = target.type(torch.int64).contiguous()  # _     (B,T) -> (B*T)

        loss += args.model.token.bolt_existing_loss * F.cross_entropy(logits, target, reduction='mean')

    if not args.model.token.bolt_type_loss == 0:
        logits = bolt_type_pred.permute(0, 2, 1)
        logits = logits.view(-1, args.num_segmentation_type).contiguous()
        # _                                                 (B,bolt_type,T) -> (B*T,bolt_type)
        target = ground_truth[:, 2:2 + args.model.token.bolt_type].view(-1)
        target = target.type(torch.int64).contiguous()  # _     (B,T) -> (B*T)

        loss += args.model.token.bolt_type_loss * F.cross_entropy(logits, target, reduction='mean')
    if not args.model.token.bolt_centers_loss == 0:
        logits = bolt_centers
        target = ground_truth[:, 2 + args.model.token.bolt_type:2 + args.model.token.bolt_type + 3]

        pred = F.sigmoid(logits)
        loss += args.model.token.bolt_centers_loss * F.huber_loss(pred, target, delta=1)
    if not args.model.token.bolt_normals_loss == 0:
        logits = bolt_normals
        target = ground_truth[:, 2 + args.model.token.bolt_type + 3:2 + args.model.token.bolt_type + 3 + 3]

        pred = F.sigmoid(logits)
        loss += args.model.token.bolt_normals_loss * F.huber_loss(pred, target, delta=1)

    return loss


def cal_segment_loss(pred, target, weights, args):
    """
        Calculate cross entropy loss, apply label smoothing if needed.
        pred: (B, segment_type, N)
        target: (B, N)
    """

    pred = pred.permute(0, 2, 1)
    pred = pred.view(-1, args.num_segmentation_type).contiguous()
    # _                                                     (B,segment_type,N) -> (B*N,segment_type)
    target = target.view(-1)
    target = target.type(torch.int64).contiguous()  # _     (B,N) -> (B*N)

    if args.use_class_weight == 0:
        loss = F.cross_entropy(pred, target, reduction='mean')
    else:
        loss = F.cross_entropy(pred, target, weight=weights, reduction='mean')

    return loss


def cal_loss(point_segmentation_pred, target, weights, transform_matrix,
             bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals,
             args):
    segment_loss = cal_segment_loss(point_segmentation_pred, target, weights, args)
    token_loss = cal_token_loss(bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals, args)

    loss = segment_loss + token_loss

    if not args.stn_loss_weight == 0:
        loss += feature_transform_reguliarzer(transform_matrix) * args.stn_loss_weight

    return loss


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    for b in range(B):
        pc = batch_data[b]
        centroid = torch.mean(pc, dim=0, keepdim=True)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1, keepdim=True)))
        pc = pc / m
        batch_data[b] = pc
    return batch_data


def rotate_per_batch(data, goals, angle_clip=np.pi * 1):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BXNx6 array, original batch of point clouds and point normals
        Return:
          BXNx3 array, rotated batch of point clouds
    """
    if goals != None:
        data = data.float()
        goals = goals.float()
        rotated_data = torch.zeros(data.shape, dtype=torch.float32)
        rotated_data = rotated_data.cuda()

        rotated_goals = torch.zeros(goals.shape, dtype=torch.float32).cuda()
        batch_size = data.shape[0]
        rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=torch.float32).cuda()
        for k in range(data.shape[0]):
            angles = []
            for i in range(3):
                angles.append(random.uniform(-angle_clip, angle_clip))
            angles = np.array(angles)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            R = torch.from_numpy(R).float().cuda()
            rotated_data[k, :, :] = torch.matmul(data[k, :, :], R)
            rotated_goals[k, :, :] == torch.matmul(goals[k, :, :], R)
            rotation_matrix[k, :, :] = R
        return rotated_data, rotated_goals, rotation_matrix
    else:
        data = data.float()
        rotated_data = torch.zeros(data.shape, dtype=torch.float32)
        rotated_data = rotated_data.cuda()

        batch_size = data.shape[0]
        rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=torch.float32).cuda()
        for k in range(data.shape[0]):
            angles = []
            for i in range(3):
                angles.append(random.uniform(-angle_clip, angle_clip))

            angles = np.array(angles)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            R = torch.from_numpy(R).float().cuda()

            rotated_data[k, :, :] = torch.matmul(data[k, :, :], R)

            rotation_matrix[k, :, :] = R
        return rotated_data, rotation_matrix


def feature_transform_reguliarzer(trans, GT=None):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    if GT == None:
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    else:
        loss = torch.mean(torch.norm(trans - GT, dim=(1, 2)))
    return loss


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


def knn(x, k):
    """
    Input:
        points: input points data, [B, N, C]
    Return:
        idx: sample index data, [B, N, K]
    """
    # x=x.permute(0,2,1)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)
    elif type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)


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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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


class SA_Layers(nn.Module):
    def __init__(self, n_layers, encoder_layer):
        super(SA_Layers, self).__init__()
        self.num_layers = n_layers
        self.encoder_layer = encoder_layer
        self.layers = _get_clones(self.encoder_layer, self.num_layers)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


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
        self.layers = _get_clones(decoder_layer, num_layers)  # repeat the decoder layers
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
