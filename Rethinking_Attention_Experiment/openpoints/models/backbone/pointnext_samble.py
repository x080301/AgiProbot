"""Official implementation of PointNext
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
https://arxiv.org/abs/2206.04670
Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
"""
from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
import math
import einops
import torch.nn.functional as F


def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set 
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):  # #layers in each blocks
            convs.append(create_convblock2d(channels[i], channels[i + 1],
                                            norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels) - 2) and not last_act else act_args,
                                            **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # neighborhood_features
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_feautres(p, dp, f, fj, self.feature_type)
        f = self.pool(self.convs(fj))
        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f


class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 **kwargs,
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        create_conv = create_convblock1d if is_head else create_convblock2d
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_conv(channels[i], channels[i + 1],
                                     norm_args=norm_args if not is_head else None,
                                     act_args=None if i == len(channels) - 2
                                                      and (self.use_res or is_head) else act_args,
                                     **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]

            # if sampler.lower() == 'fps':
            #     self.sample_fn = furthest_point_sample
            # elif sampler.lower() == 'random':
            #     self.sample_fn = random_sample
            self.samble_downsample = DownSampleToken(self.stride, in_channels)

    def forward(self, pf):
        p, f = pf

        # print(f"p.shape:{p.shape}")
        # print(f"f.shape:{f.shape}")

        if self.is_head:
            f = self.convs(f)  # (n, c)
        else:
            if not self.all_aggr:

                # idx = self.sample_fn(p, p.shape[1] // self.stride).long()

                f, idx = self.samble_downsample(f)
                f = f.contiguous()

                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None

            # print(f"new_p.shape: {new_p.shape}")

            dp, fj = self.grouper(new_p, p, f)
            fj = get_aggregation_feautres(new_p, dp, fi, fj, feature_type=self.feature_type)
            f = self.pool(self.convs(fj))
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation([in_channels, in_channels, mid_channels, in_channels],
                                      norm_args=norm_args, act_args=None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


@MODELS.register_module()
class PointNextEncoder_SAMBLE(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'})
        self.act_args = kwargs.get('act_args', {'act': 'relu'})
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, **self.aggr_args
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)


@MODELS.register_module()
class PointNextDecoder_SAMBLE(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        return f[-len(self.decoder) - 1]


@MODELS.register_module()
class PointNextPartDecoder_SAMBLE(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_blocks: List[int] = [1, 1, 1, 1],
                 decoder_strides: List[int] = [4, 4, 4, 4],
                 act_args: str = 'relu',
                 cls_map='pointnet2',
                 num_classes: int = 16,
                 cls2partembed=None,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        fp_channels = encoder_channel_list[:-1]

        # the following is for decoder blocks
        self.conv_args = kwargs.get('conv_args', None)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)
        block = kwargs.get('block', 'InvResMLP')
        if isinstance(block, str):
            block = eval(block)
        self.blocks = decoder_blocks
        self.strides = decoder_strides
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'})
        self.act_args = kwargs.get('act_args', {'act': 'relu'})
        self.expansion = kwargs.get('expansion', 4)
        radius = kwargs.get('radius', 0.1)
        nsample = kwargs.get('nsample', 16)
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        self.cls_map = cls_map
        self.num_classes = num_classes
        self.use_res = kwargs.get('use_res', True)
        group_args = kwargs.get('group_args', {'NAME': 'ballquery'})
        self.aggr_args = kwargs.get('aggr_args',
                                    {'feature_type': 'dp_fj', "reduction": 'max'}
                                    )
        if self.cls_map == 'curvenet':
            # global features
            self.global_conv2 = nn.Sequential(
                create_convblock1d(fp_channels[-1] * 2, 128,
                                   norm_args=None,
                                   act_args=act_args))
            self.global_conv1 = nn.Sequential(
                create_convblock1d(fp_channels[-2] * 2, 64,
                                   norm_args=None,
                                   act_args=act_args))
            skip_channels[0] += 64 + 128 + 16  # shape categories labels
        elif self.cls_map == 'pointnet2':
            self.convc = nn.Sequential(create_convblock1d(16, 64,
                                                          norm_args=None,
                                                          act_args=act_args))
            skip_channels[0] += 64  # shape categories labels

        elif self.cls_map == 'pointnext':
            self.global_conv2 = nn.Sequential(
                create_convblock1d(fp_channels[-1] * 2, 128,
                                   norm_args=None,
                                   act_args=act_args))
            self.global_conv1 = nn.Sequential(
                create_convblock1d(fp_channels[-2] * 2, 64,
                                   norm_args=None,
                                   act_args=act_args))
            skip_channels[0] += 64 + 128 + 50  # shape categories labels
            self.cls2partembed = cls2partembed
        elif self.cls_map == 'pointnext1':
            self.convc = nn.Sequential(create_convblock1d(50, 64,
                                                          norm_args=None,
                                                          act_args=act_args))
            skip_channels[0] += 64  # shape categories labels
            self.cls2partembed = cls2partembed

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i], group_args=group_args, block=block, blocks=self.blocks[i])

        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels, group_args=None, block=None, blocks=1):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp, act_args=self.act_args))
        self.in_channels = fp_channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def forward(self, p, f, cls_label):
        B, N = p[0].shape[0:2]
        if self.cls_map == 'curvenet':
            emb1 = self.global_conv1(f[-2])
            emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
            emb2 = self.global_conv2(f[-1])
            emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1
            cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
            cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1)
            cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
            cls_one_hot = cls_one_hot.expand(-1, -1, N)
        elif self.cls_map == 'pointnet2':
            cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
            cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1).repeat(1, 1, N)
            cls_one_hot = self.convc(cls_one_hot)
        elif self.cls_map == 'pointnext':
            emb1 = self.global_conv1(f[-2])
            emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
            emb2 = self.global_conv2(f[-1])
            emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1
            self.cls2partembed = self.cls2partembed.to(p[0].device)
            cls_one_hot = self.cls2partembed[cls_label.squeeze()].unsqueeze(-1)
            cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
            cls_one_hot = cls_one_hot.expand(-1, -1, N)
        elif self.cls_map == 'pointnext1':
            self.cls2partembed = self.cls2partembed.to(p[0].device)
            cls_one_hot = self.cls2partembed[cls_label.squeeze()].unsqueeze(-1).expand(-1, -1, N)
            cls_one_hot = self.convc(cls_one_hot)

        for i in range(-1, -len(self.decoder), -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i - 1], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

        # TODO: study where to add this ? 
        f[-len(self.decoder) - 1] = self.decoder[0][1:](
            [p[1], self.decoder[0][0]([p[1], torch.cat([cls_one_hot, f[1]], 1)], [p[2], f[2]])])[1]

        return f[-len(self.decoder) - 1]


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
        # x.shape == (B, C, N)

        B, C, N = x.shape
        M = N // 2

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

        if torch.sum(max_num_points) == 0:
            print("No points found")
            print(f'max_num_points:{max_num_points}')
            print(f'self.bin_boundaries:{self.bin_boundaries}')
            print(f'self.attention_point_score:{self.attention_point_score}')
            exit(-1)

        # max_num_points:(B,num_bins)
        self.k_point_to_choose = calculate_num_points_to_choose(M, bin_weights, max_num_points, self.stride)
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
            M,
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

        for i in range(attention_point_score.shape[0]):
            if torch.sum(attention_point_score[i])==0:
                print(f'torch.sum(sparse_attention_map, dim=-2)[i]:{torch.sum(sparse_attention_map, dim=-2)[i]}')
                exit(-1)

        # if torch.isnan(attention_point_score).any():
        #     print(f'attention_point_score:{attention_point_score}')
        #     print(f'sparse_num:{sparse_num}')
        #     print(f'sparse_attention_map:{sparse_attention_map}')
        attention_point_score[torch.isnan(attention_point_score)] = 0

        return attention_point_score, sparse_attention_map, mask


def calculate_num_points_to_choose(total_points_to_choose, bin_prob, max_num_points, stride):
    """

    :param total_points_to_choose: Int
    :param bin_prob: torch.Tensor(B,num_bins)
    :param max_num_points: torch.Tensor(B,num_bins)
    :return: number of choosen points, torch.Tensor(B,num_bins)
    """

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

    # print(
    #     f"..........{num_chosen_points_in_bin[torch.arange(0, B), torch.argmax(max_num_points - num_chosen_points_in_bin, dim=1)].shape}")
    # print(f'total_points_to_choose.shape:{total_points_to_choose.shape}')
    # print(f'torch.sum(num_chosen_points_in_bin, dim=1).shape:{torch.sum(num_chosen_points_in_bin, dim=1).shape}')
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


def generating_downsampled_index(M, attention_point_score, bin_points_mask, bin_sample_mode, boltzmann_t,
                                 k_point_to_choose):
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

        try:
            sampled_index_M_points = torch.multinomial(sampling_probabilities, M)
            # sampled_index_M_points: (B*num_bins,M)
            sampled_index_M_points = sampled_index_M_points.reshape(B, num_bins, M)
            # sampled_index_M_points: (B,num_bins,M)
        except:
            print(f'M:{M}')
            print(f'k_point_to_choose[0, :]:{k_point_to_choose[0, :]}')
            print(f'k_point_to_choose:{k_point_to_choose}')

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

    # attention_point_score: (B,1,N)
    attention_point_score = (attention_point_score - torch.mean(attention_point_score, dim=2, keepdim=True)) \
                            / torch.std(attention_point_score, dim=2, unbiased=False, keepdim=True)+1e-8

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
