import torch
from torch import nn
from utils import ops
import math
import torch.nn.functional as F
import einops

from utils.ops import (
    calculate_num_points_to_choose,
    bin_partition,
    generating_downsampled_index,
)


def bin_probability_multiple(
        x_ds,
        input_x_shape,
        down_sampling_idx,
        bin_chunks_idx,
        bin_probability,
        direct_link_mode,
):
    B, C, N = input_x_shape
    _, _, M = x_ds.shape
    num_bins = len(bin_chunks_idx)
    # bin_prob.shape == (B, num_bins)
    bin_probability = bin_probability / torch.sum(bin_probability, dim=1, keepdim=True)
    if (
            direct_link_mode == "no_link"
            or direct_link_mode == "no_link_no_sigmoid"
            or direct_link_mode == "raw_gradient"
    ):
        bin_probability = bin_probability / M + 1.0
    elif (
            direct_link_mode == "no_link_higher_gradient"
            or direct_link_mode == "higher_gradient"
    ):
        bin_probability = (
                bin_probability - bin_probability.clone().detach() * (M - 1) / M + 1.0
        )
    else:
        raise NotImplementedError

    assert down_sampling_idx.shape[1] == 1, "Number of heads should be 1!"

    tensor_to_multiply = torch.zeros(B, N).to(bin_probability.device)

    for i in range(num_bins):
        for j in range(B):
            # print(f'tensor_to_multiply[j, :].shape:{tensor_to_multiply[j, :].shape}')
            # print(f'bin_chunks_idx[i].squeeze()[j, :]:{bin_chunks_idx[i].squeeze()[j, :].shape}')
            # print(f'bin_probability[j, i]:{bin_probability[j, i]}')
            tensor_to_multiply[j, :][bin_chunks_idx[i][j].flatten()] = bin_probability[
                j, i
            ]

    tensor_to_multiply = torch.gather(
        tensor_to_multiply, dim=1, index=down_sampling_idx.squeeze(dim=1)
    ).unsqueeze(1)

    # print(f'x_ds.shape=={x_ds.shape}')
    # print(f'tensor_to_multiply:{tensor_to_multiply.shape}')
    x_ds = x_ds * tensor_to_multiply

    return x_ds


def calculate_num_points_to_choose_one_iteration(
        probability, max_num_points, num_undecided_points, num_points_in_bins
):  # , total_points):
    """

    :param num_undecided_points: torch.Tensor(B,)
    :param probability: torch.Tensor(B,num_bins)
    :param max_num_points: torch.Tensor(B,num_bins)
    :return: number of choosen points, torch.Tensor(B,num_bins);
    """
    num_undecided_points = num_undecided_points.reshape(-1, 1)

    # probability = probability / torch.sum(probability, dim=1, keepdim=True) * num_undecided_points / torch.sum(
    #     max_num_points, dim=1, keepdim=True)

    # print(f'max_num_points{max_num_points}')
    # print(f'probability{probability}')
    num_points_to_choose = probability * num_points_in_bins
    num_points_to_choose = (
            num_points_to_choose
            * num_undecided_points
            / torch.sum(num_points_to_choose, dim=1, keepdim=True)
    )
    # print(f'num_points_to_choose2:{num_points_to_choose}')
    num_points_to_choose = num_points_to_choose.int()

    # print(f'num_points_to_choose1:{num_points_to_choose}')
    # num_points_to_choose = num_points_to_choose * total_points / torch.sum(num_points_to_choose, dim=1, keepdim=True)
    num_points_to_choose = torch.where(
        num_points_to_choose < max_num_points, num_points_to_choose, max_num_points
    )
    # print(f'num_points_to_choose:{num_points_to_choose}')

    return num_points_to_choose


def nonuniform_bin_idx_selection(
        attention_point_score,
        bin_boundaries,
        bin_prob,
        normalization_mode,
        M,
        bin_sample_mode,
        dynamic_boundaries_enable,
):
    B, H, N = attention_point_score.shape
    _, num_bins = bin_prob.shape

    bin_prob = F.relu(bin_prob)  # .clone().detach())
    # bin_prob.shape == (B, num_bins)
    # self.attention_point_score.shape == (B, H, N)

    aps_chunks, idx_chunks, bin_boundaries, _ = ops.sort_chunk_nonuniform(
        attention_point_score,
        bin_boundaries,
        num_bins,
        normalization_mode,
        dynamic_boundaries_enable,
    )

    # print(f'len(idx_chunks):{len(aps_chunks)}')
    # print(f'len(idx_chunks[0]):{len(aps_chunks[0])}')
    # print(f'len(idx_chunks[0]):{aps_chunks[0][0].shape}')
    # print(f'idx.dtype3:{idx_chunks[0][0].dtype}')
    # aps_chunks.shape == num_bins * (B, H, n)
    # idx_chunks.shape == num_bins * (B, H, n)

    # chunk_size = aps_chunks[j][i].shape[1]
    assert H == 1, "Number of heads should be 1!"

    max_num_points = torch.zeros(
        (B, num_bins), dtype=torch.long, device=bin_prob.device
    )
    for i in range(B):
        for j in range(num_bins):
            max_num_points[i, j] = aps_chunks[j][i].shape[1]
    # print(f' bin_prob{bin_prob}-----------')

    k_point_to_choose = calculate_num_points_to_choose(bin_prob, max_num_points, M)
    # k_point_to_choose.shape == (B, num_bins)

    # print(f'k_point_to_choose{torch.sum(k_point_to_choose,dim=1)}')

    idx_batch_list = []
    for i in range(B):

        idx_list = []
        # print(f'sampling_scale:{sampling_scale}')
        # print(f'self.M:{self.M}')
        # print(f'chunk_size_list:{chunk_size_list}')

        for j in range(num_bins):
            # each bin has k samples
            k = k_point_to_choose[i, j]

            if bin_sample_mode == "topk":
                idx_tmp = aps_chunks[j][i].topk(k, dim=-1)[1]  # idx.shape == (H, k)
            elif bin_sample_mode == "uniform":
                idx_tmp = torch.randperm(aps_chunks[j][i].shape[1])[:k]
                idx_tmp = (
                    idx_tmp.unsqueeze(0).expand(H, -1).to(attention_point_score.device)
                )
            elif bin_sample_mode == "random":
                if k != 0:
                    # aps_chunks_tmp = ops.norm_range(aps_chunks[j][i], dim=-1, n_min=0, n_max=1, mode="minmax")
                    # aps_chunks_tmp = torch.nn.functional.softmax(aps_chunks_tmp, dim=-1)
                    aps_chunks_tmp = torch.nn.functional.softmax(
                        aps_chunks[j][i], dim=-1
                    )
                    # print(f'k:{k}')
                    # print(f'aps_chunks_tmp.shape:{aps_chunks_tmp.shape}')
                    if aps_chunks_tmp.nelement() < k:
                        print(f"aps_chunks_tmp{aps_chunks_tmp.nelement()},k{k}")
                        exit(-1)
                    idx_tmp = torch.multinomial(
                        aps_chunks_tmp, num_samples=k, replacement=False
                    )
            else:
                raise ValueError(
                    "Please check the setting of bin sample mode. It must be topk, multinomial or random!"
                )
            # print(f'k:{k}')
            # print(f'idx_tmp:{idx_tmp}')
            # print(f'idx_tmp.shape:{idx_tmp.shape}')
            # print(f'idx_chunks[j][i].shape:{idx_chunks[j][i].shape}')
            if k != 0:
                idx = idx_chunks[j][i][0, idx_tmp[0]].reshape(1, -1)
                # torch.gather(idx_chunks[j][i], dim=-1, index=idx_tmp)  # idx.shape == (H, k)
                idx_list.append(idx)
        idx_single = torch.cat(idx_list, dim=-1)  # idx_list.shape == (H, M)
        idx_batch_list.append(idx_single)
    idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)

    # k_point_to_choose.shape == (B, num_bins)
    # idx_chunks.shape == num_bins * (B, H, n)
    return idx_batch, k_point_to_choose, idx_chunks, bin_boundaries


def nonuniform_bin_idx_selection_beforesoftmaxbinprob(
        attention_point_score,
        bin_boundaries,
        attention_bins_beforesoftmax,
        normalization_mode,
        M,
        bin_sample_mode,
        dynamic_boundaries_enable,
        relu_mean_order,
        num_bins,
        momentum_update_factor,
        boltzmann_T,
):
    # attention_bins_beforesoftmax: (B,1,N,num_bins) or (B,1,N,1)
    B, H, N, _ = attention_bins_beforesoftmax.shape

    if attention_bins_beforesoftmax.shape[3] == 1:
        attention_bins_beforesoftmax = einops.repeat(
            attention_bins_beforesoftmax, "b 1 n 1 -> b 1 n d", d=num_bins
        )

    assert H == 1, "Number of heads should be 1!"

    aps_chunks, idx_chunks, bin_boundaries, bin_points_mask = ops.sort_chunk_nonuniform(
        attention_point_score,
        bin_boundaries,
        num_bins,
        normalization_mode,
        dynamic_boundaries_enable,
        momentum_update_factor,
    )
    # aps_chunks.shape == num_bins * (B, H, n)
    # idx_chunks.shape == num_bins * (B, H, n)
    # bin_points_mask: (B,H,N,num_bins)

    masked_attention_map_token = attention_bins_beforesoftmax * bin_points_mask
    if relu_mean_order == "mean_relu":

        bin_prob_return = torch.sum(masked_attention_map_token, dim=2) / (
                torch.count_nonzero(bin_points_mask, dim=2) + 1e-8
        )
        # torch.count_nonzero(masked_attention_map_token, dim=2) + 1e-8)
        bin_prob_return = bin_prob_return.squeeze(1)
        bin_prob = F.relu(bin_prob_return)
    elif relu_mean_order == "relu_mean":
        masked_attention_map_token = F.relu(masked_attention_map_token)
        bin_prob_return = torch.sum(masked_attention_map_token, dim=2) / (
                torch.count_nonzero(bin_points_mask, dim=2) + 1e-8
        )
        bin_prob_return = bin_prob_return.squeeze(1)
        bin_prob = bin_prob_return
    else:
        raise NotImplementedError

    # bin_prob.shape == (B, num_bins)

    # chunk_size = aps_chunks[j][i].shape[1]

    max_num_points = torch.zeros(
        (B, num_bins), dtype=torch.long, device=bin_prob.device
    )
    # max_num_points:(B,num_bins)
    for i in range(B):
        for j in range(num_bins):
            max_num_points[i, j] = aps_chunks[j][i].shape[1]
    # print(f' bin_prob{bin_prob}-----------')

    k_point_to_choose = calculate_num_points_to_choose(bin_prob, max_num_points, M)
    # k_point_to_choose.shape == (B, num_bins)

    idx_batch_list = []
    for i in range(B):

        idx_list = []
        # print(f'sampling_scale:{sampling_scale}')
        # print(f'self.M:{self.M}')
        # print(f'chunk_size_list:{chunk_size_list}')

        for j in range(num_bins):
            # each bin has k samples
            k = k_point_to_choose[i, j]

            if bin_sample_mode == "topk":
                idx_tmp = aps_chunks[j][i].topk(k, dim=-1)[1]  # idx.shape == (H, k)
            elif bin_sample_mode == "uniform":
                idx_tmp = torch.randperm(aps_chunks[j][i].shape[1])[:k]
                idx_tmp = (
                    idx_tmp.unsqueeze(0).expand(H, -1).to(attention_point_score.device)
                )
            elif bin_sample_mode == "random":
                if k != 0:
                    # aps_chunks_tmp = ops.norm_range(aps_chunks[j][i], dim=-1, n_min=0, n_max=1, mode="minmax")
                    # aps_chunks_tmp = torch.nn.functional.softmax(aps_chunks_tmp, dim=-1)
                    aps_chunks_tmp = torch.nn.functional.softmax(
                        aps_chunks[j][i], dim=-1
                    )
                    # print(f'k:{k}')
                    # print(f'aps_chunks_tmp.shape:{aps_chunks_tmp.shape}')
                    if aps_chunks_tmp.nelement() < k:
                        print(
                            f"bin_num:{torch.count_nonzero(masked_attention_map_token, dim=2).squeeze(1)}"
                        )
                        print(f"k_point_to_choose:{k_point_to_choose}")
                        print(f"bin_prob:{bin_prob}")

                        print(f"aps_chunks_tmp:{aps_chunks_tmp.nelement()},k:{k}")
                        exit(-1)
                    # nan_inf_negative_mask = ((aps_chunks_tmp == float('inf'))
                    #                          | torch.isnan(aps_chunks_tmp))
                    # aps_chunks_tmp = torch.where(nan_inf_negative_mask, 0, aps_chunks_tmp)

                    if (
                            torch.count_nonzero(aps_chunks_tmp == float("inf"))
                            + torch.count_nonzero(aps_chunks_tmp < 0)
                            + torch.count_nonzero(torch.isnan(aps_chunks_tmp))
                            != 0
                    ):
                        print(
                            f'\nnum inf elements:{torch.count_nonzero(aps_chunks_tmp == float("inf"))}\n'
                        )
                        print(
                            f"\nnum negative elements:{torch.count_nonzero(aps_chunks_tmp < 0)}\n"
                        )
                        print(
                            f"\nnum nan elements:{torch.count_nonzero(torch.isnan(aps_chunks_tmp))}\n"
                        )

                    idx_tmp = torch.multinomial(
                        aps_chunks_tmp / boltzmann_T, num_samples=k, replacement=False
                    )

            else:
                raise ValueError(
                    "Please check the setting of bin sample mode. It must be topk, multinomial or random!"
                )
            # print(f'k:{k}')
            # print(f'idx_tmp:{idx_tmp}')
            # print(f'idx_tmp.shape:{idx_tmp.shape}')
            # print(f'idx_chunks[j][i].shape:{idx_chunks[j][i].shape}')
            if k != 0:
                idx = idx_chunks[j][i][0, idx_tmp[0]].reshape(1, -1)
                # torch.gather(idx_chunks[j][i], dim=-1, index=idx_tmp)  # idx.shape == (H, k)
                idx_list.append(idx)
        idx_single = torch.cat(idx_list, dim=-1)  # idx_list.shape == (H, M)
        idx_batch_list.append(idx_single)
    idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)

    # k_point_to_choose.shape == (B, num_bins)
    # idx_chunks.shape == num_bins * (B, H, n)
    return idx_batch, k_point_to_choose, idx_chunks, bin_boundaries, bin_prob_return


class DownSampleCarve(nn.Module):
    def __init__(self, config_ds, layer):
        super(DownSampleCarve, self).__init__()

        self.M = config_ds.M[layer]
        self.K = config_ds.K
        self.asm = config_ds.asm[layer]
        self.res = config_ds.res.enable[layer]
        self.ff = config_ds.res.ff[layer]
        self.num_heads = config_ds.num_heads[layer]
        self.idx_mode = config_ds.idx_mode[layer]
        q_in = config_ds.q_in[layer]
        q_out = config_ds.q_out[layer]
        k_in = config_ds.k_in[layer]
        k_out = config_ds.k_out[layer]
        v_in = config_ds.v_in[layer]
        v_out = config_ds.v_out[layer]

        self.q_depth = int(q_out / self.num_heads)
        self.k_depth = int(k_out / self.num_heads)
        self.v_depth = int(v_out / self.num_heads)
        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        # downsample res link
        if self.res:
            self.bn1 = nn.BatchNorm1d(v_out)
            if self.ff:
                self.ffn = nn.Sequential(
                    nn.Conv1d(128, 512, 1, bias=False),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Conv1d(512, 128, 1, bias=False),
                )
                self.bn2 = nn.BatchNorm1d(v_out)
        # bin
        self.bin_enable = config_ds.bin.enable[layer]
        self.num_bins = config_ds.bin.num_bins[layer]
        self.scaling_factor = config_ds.bin.scaling_factor[layer]
        self.bin_sample_mode = config_ds.bin.sample_mode[layer]
        self.bin_norm_mode = config_ds.bin.norm_mode[layer]
        self.bin_mode = config_ds.bin.mode[layer]
        self.enable_multiply = config_ds.bin.multiply[layer]
        self.direct_link_mode = config_ds.bin.direct_link_mode[layer]

        if self.bin_enable:
            if self.bin_mode == "mode1":
                self.bin_conv1 = nn.Conv1d(q_in, int(self.num_bins / 2), 1, bias=False)
                self.bin_conv2 = nn.Conv1d(
                    q_in + int(self.num_bins / 2), q_out, 1, bias=False
                )
            elif self.bin_mode == "nonuniform_split_bin":
                if "no_link" in self.direct_link_mode:
                    self.bin_conv1 = nn.Conv1d(q_in, int(self.num_bins), 1, bias=False)
                else:
                    self.bin_conv1 = nn.Conv1d(q_in, int(self.num_bins), 1, bias=False)
                    self.bin_conv2 = nn.Conv1d(
                        q_in + int(self.num_bins), q_out, 1, bias=False
                    )

                # self.bin_boundaries = config_ds.bin.bin_boundaries[layer]
                self.dynamic_boundaries = config_ds.bin.dynamic_boundaries
                if config_ds.bin.dynamic_boundaries:
                    self.bin_boundaries = None
                else:
                    bin_boundaries_upper = [float("inf")]
                    bin_boundaries_upper.extend(config_ds.bin.bin_boundaries[layer])
                    bin_boundaries_lower = config_ds.bin.bin_boundaries[layer]
                    bin_boundaries_lower.append(float("-inf"))
                    self.bin_boundaries = [
                        torch.asarray(bin_boundaries_upper).reshape(
                            1, 1, 1, self.num_bins
                        ),
                        # [inf, 0.503, 0.031, -0.230, -0.427, -0.627]
                        torch.asarray(bin_boundaries_lower).reshape(
                            1, 1, 1, self.num_bins
                        ),
                        # [0.503, 0.031, -0.230, -0.427, -0.627, -inf]
                    ]

                self.normalization_mode = config_ds.bin.normalization_mode[layer]
            else:
                raise NotImplementedError

        # boltzmann
        self.boltzmann_enable = config_ds.boltzmann.enable[layer]
        self.boltzmann_T = config_ds.boltzmann.boltzmann_T[layer]
        self.boltzmann_norm_mode = config_ds.boltzmann.norm_mode[layer]
        # positional_encoding
        self.pe = config_ds.pe.enable[layer]
        self.pe_mode = config_ds.pe.mode[layer]
        if self.pe:
            if self.pe_mode == "III":
                self.q_pe_conv = nn.Conv1d(3, q_out, 1, bias=False)
                self.v_pe_conv = nn.Conv1d(3, q_out, 1, bias=False)
            elif self.pe_mode == "IV":
                self.q_pe_conv = nn.Conv1d(3, q_out, 1, bias=False)
                self.k_pe_conv = nn.Conv1d(3, q_out, 1, bias=False)
                self.v_pe_conv = nn.Conv1d(3, q_out, 1, bias=False)
            else:
                raise ValueError(f"pe_mode must be III or IV, Got{self.pe_mode}!")

    def forward(self, x, x_xyz=None):
        # x.shape == (B, C, N)
        # if self.bin_enable and self.bin_mode == "mode1":
        if self.bin_enable:
            if self.bin_mode == "mode1" or self.bin_mode == "nonuniform_split_bin":
                x, bin_prob = self.bin_conv(x, bin_mode=self.bin_mode)
                # bin_prob.shape == (B, num_bins)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, N)
        k = self.k_conv(x)
        # k.shape ==  (B, C, N)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, N)
        v = self.v_conv(x)
        # v.shape ==  (B, C, N)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, N)
        q = q.permute(0, 1, 3, 2)  # q.shape == (B, H, N, D)
        if self.pe:
            q_pe = self.q_pe_conv(x_xyz)
            q_pe = self.split_heads(
                q_pe, self.num_heads, self.q_depth
            )  # q_pe.shape == (B, H, D, N)
            v_pe = self.v_pe_conv(x_xyz)
            v_pe = self.split_heads(
                v_pe, self.num_heads, self.v_depth
            )  # v_pe.shape == (B, H, D, N)
            if self.pe_mode == "IV":
                k_pe = self.k_pe_conv(x_xyz)
                k_pe = self.split_heads(
                    k_pe, self.num_heads, self.k_depth
                )  # k_pe.shape == (B, H, D, N)
                self.k_pe = (
                        k.permute(0, 1, 3, 2) @ k_pe
                )  # self.k_pe.shape == (B, H, N, N)
            self.q_pe = q @ q_pe  # self.q_pe.shape == (B, H, N, N)
            v = v + v_pe  # v.shape == (B, H, D, N)

        self.attention_map = self.attention_scoring(
            q, k
        )  # self.attention_map.shape == (B, H, N, N)

        idx, self.attention_point_score, self.sparse_attention_map, self.mask = (
            self.idx_selection(x)
        )
        if self.bin_enable:
            if self.bin_mode == "mode1":
                idx, _, idx_chunks = self.bin_idx_selection(
                    self.attention_point_score, self.num_bins, bin_prob, self.M
                )
            elif self.bin_mode == "mode2":
                idx, _ = self.bin2_idx_selection()
            elif self.bin_mode == "nonuniform_split_bin":
                bin_boundaries = [item.to(x.device) for item in self.bin_boundaries]
                idx, k_point_to_choose, idx_chunks, _ = nonuniform_bin_idx_selection(
                    self.attention_point_score,
                    bin_boundaries,
                    bin_prob,
                    self.normalization_mode,
                    self.M,
                    self.bin_sample_mode,
                    self.dynamic_boundaries,
                )
                # k_point_to_choose.shape == (B, num_bins)
                # idx_chunks.shape == num_bins * (B, H, n)

            else:
                raise NotImplementedError

        elif self.boltzmann_enable:
            idx = self.boltzmann_idx_selection(
                self.attention_point_score,
                self.M,
                self.boltzmann_norm_mode,
                self.boltzmann_T,
            )

        # idx_dropped = torch.sum(self.attention_map, dim=-2).topk(self.attention_map.shape[-1] - self.M, dim=-1, largest=False)[1]
        # idx_dropped.shape == (B, H, N-M)
        # print(f'idx.dtype1:{idx.dtype}')
        attention_down = torch.gather(
            self.attention_map,
            dim=2,
            index=idx.unsqueeze(3).expand(-1, -1, -1, self.attention_map.shape[-1]),
        )
        # attention_down.shape == (B, H, M, N)
        # attention_dropped = torch.gather(self.attention_map, dim=2,
        #                                  index=idx_dropped[..., None].expand(-1, -1, -1, k.shape[-1]))
        # attention_dropped.shape == (B, H, N-M, N)
        v_down = (attention_down @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # v_down.shape == (B, M, H, D)
        # v_dropped = (attention_dropped @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # v_dropped.shape == (B, N-M, H, D)
        x_ds = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)
        # v_down.shape == (B, C, M)

        # residual & feedforward
        if self.res is True:
            x_ds = self.res_block(x, x_ds, idx)

        # x_dropped = v_dropped.reshape(v_dropped.shape[0], v_dropped.shape[1], -1).permute(0, 2, 1)
        # v_dropped.shape == (B, C, N-M)
        # return (x_ds, idx), (x_dropped, idx_dropped)

        if self.enable_multiply:
            x_ds = bin_probability_multiple(
                x_ds, x.shape, idx, idx_chunks, bin_prob, self.direct_link_mode
            )

        self.idx = idx
        self.idx_chunks = idx_chunks
        # idx_chunks.shape == num_bins * (B, H, n)
        self.bin_prob = bin_prob
        self.k_point_to_choose = k_point_to_choose
        # k_point_to_choose.shape == (B, num_bins)
        return (x_ds, idx), (None, None)

    def output_variables(self, *args):

        # print(vars().keys())
        variables = None
        for i, key in enumerate(args):
            if i == 0:
                variables = getattr(vars()["self"], key)
                # variables = vars()[f'self.{key}']
            elif i == 1:
                variables = (variables,) + (getattr(vars()["self"], key),)
                # variables = (variables,) + (vars()[f'self.{key}'],)
            else:
                variables = variables + (getattr(vars()["self"], key),)
                # variables = variables + (vars()[f'self.{key}'],)

        return variables

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x

    def attention_scoring(
            self, q, k
    ):  # q.shape == (B, H, N, D), k.shape == (B, H, D, N)
        if self.asm == "dot":
            energy = q @ k  # energy.shape == (B, H, N, N)
        elif self.asm == "l2":
            energy = -1 * ops.l2_global(q, k)  # -(Q-K)^2 energy.shape == (B, H, N, N)
        elif self.asm == "l2+":
            energy = ops.l2_global(q, k)  # (Q-K)^2 energy.shape == (B, H, N, N)
        else:
            raise ValueError("Please check the setting of asm!")
        if self.pe:
            if self.pe_mode == "III":
                energy = energy + self.q_pe  # energy.shape == (B, H, N, N)
            elif self.pe_mode == "IV":
                energy = energy + self.q_pe + self.k_pe  # energy.shape == (B, H, N, N)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(
            energy / scale_factor
        )  # attention.shape == (B, H, N, N)
        return attention

    def res_block(self, x, x_ds, idx):  # x.shape == (B, C, N), x_ds.shape == (B, C, M)
        x_tmp = torch.gather(x, dim=-1, index=idx)  # x_res.shape == (B, 1, M)
        x_res = self.bn1(x_ds + x_tmp)  # x_res.shape == (B, C, M)
        if self.ff == True:
            x_tmp = self.ffn(x_res)
            x_res = self.bn2(x_ds + x_tmp)
        return x_res  # x_res.shape == (B, C, M)

    def get_sparse_attention_map(self, x):
        mask = ops.neighbor_mask(x, self.K)
        mask = mask.unsqueeze(1).expand(-1, self.attention_map.shape[1], -1, -1)
        # print(f'attention_map.shape{self.attention_map.shape}')
        # print(f'mask.shape{mask.shape}')
        # exit(-1)
        sparse_attention_map = self.attention_map * mask
        return mask, sparse_attention_map

    def idx_selection(self, x):
        mask, sparse_attention_map = self.get_sparse_attention_map(x)
        sparse_num = torch.sum(mask, dim=-2) + 1e-8

        # full attention map based
        if self.idx_mode == "col_sum":
            attention_point_score = torch.sum(
                self.attention_map, dim=-2
            )  # self.attention_point_score.shape == (B, H, N)
        elif self.idx_mode == "row_std":
            attention_point_score = torch.std(self.attention_map, dim=-1)

        # sparse attention map based

        elif self.idx_mode == "sparse_row_sum":
            attention_point_score = torch.sum(sparse_attention_map, dim=-1)
        elif self.idx_mode == "sparse_row_std":
            sparse_attention_map_std = sparse_attention_map.masked_select(
                mask != 0
            ).view(sparse_attention_map.shape[:-1] + (self.K,))
            attention_point_score = torch.std(sparse_attention_map_std, dim=-1)
        elif self.idx_mode == "sparse_col_sum":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2)
        elif self.idx_mode == "sparse_col_avg":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2) / sparse_num
        elif self.idx_mode == "sparse_col_sqr":
            attention_point_score = (
                    torch.sum(sparse_attention_map, dim=-2) / sparse_num / sparse_num
            )
        else:
            raise ValueError("Please check the setting of idx mode!")
        idx = attention_point_score.topk(self.M, dim=-1)[1]
        return idx, attention_point_score, sparse_attention_map, mask

    def bin_conv(self, x, bin_mode="mode1"):
        if bin_mode == "nonuniform_split_bin":
            if self.direct_link_mode == "raw":
                bin_prob_edge = self.bin_conv1(
                    x
                )  # bin_prob_edge.shape == (B, num_bins, N)
                x = torch.cat(
                    (x, bin_prob_edge), dim=1
                )  # x.shape == (B, C+num_bins, N)
                x = self.bin_conv2(x)  # x.shape == (B, C, N)

                bin_prob = torch.max(bin_prob_edge, dim=-1, keepdim=True)[0]
                # bin_prob_edge.shape == (B, num_bins, 1)
                # print(f'bin_prob_edge before bin_prob:{bin_prob_edge[0, :, 0]}')
                bin_prob = F.sigmoid(bin_prob).squeeze(2)
                # bin_prob = bin_prob.squeeze(2)

                # bin_prob.shape == (B, num_bins)
            elif (
                    self.direct_link_mode == "no_link"
                    or self.direct_link_mode == "no_link_higher_gradient"
            ):
                bin_prob_edge = self.bin_conv1(
                    x
                )  # bin_prob_edge.shape == (B, num_bins, N)
                bin_prob_edge = torch.max(bin_prob_edge, dim=-1, keepdim=True)[0]
                # bin_prob_edge.shape == (B, num_bins, 1)
                bin_prob = F.sigmoid(bin_prob_edge).squeeze(2)
            elif self.direct_link_mode == "no_link_no_sigmoid":
                bin_prob_edge = self.bin_conv1(
                    x
                )  # bin_prob_edge.shape == (B, num_bins, N)
                bin_prob_edge = torch.max(bin_prob_edge, dim=-1, keepdim=True)[0]
                bin_prob = bin_prob_edge.squeeze(2)
            else:
                raise NotImplementedError

        elif bin_mode == "mode1":
            bin_prob_edge = self.bin_conv1(
                x
            )  # bin_prob_edge.shape == (B, num_bins/2, N)
            x = torch.cat((x, bin_prob_edge), dim=1)  # x.shape == (B, C+num_bins/2, N)
            x = self.bin_conv2(x)  # x.shape == (B, C, N)

            bin_prob_edge = torch.max(bin_prob_edge, dim=-1, keepdim=True)[
                0
            ]  # bin_prob_edge.shape == (B, num_bins/2, 1)
            bin_prob_edge = bin_prob_edge.permute(
                0, 2, 1
            )  # bin_prob_edge.shape == (B, 1, num_bins/2)
            bin_prob_edge = bin_prob_edge / self.scaling_factor
            bin_prob_edge = ops.norm_range(
                bin_prob_edge, dim=-1, n_min=0.5, n_max=1, mode=self.bin_norm_mode
            )
            bin_prob_inner = torch.flip((1 - bin_prob_edge), dims=(-1,))
            bin_prob = torch.cat(
                (bin_prob_edge, bin_prob_inner), dim=-1
            )  # bin_prob.shape == (B, 1, num_bins)
            bin_prob = bin_prob.squeeze(1)  # bin_prob.shape == (B, num_bins)

        else:
            raise NotImplementedError

        return x, bin_prob

    def bin_idx_selection(self, attention_point_score, num_bins, bin_prob, M):
        # self.attention_point_score.shape == (B, H, N)
        aps_chunks, idx_chunks = ops.sort_chunk(
            attention_point_score, num_bins, dim=-1, descending=True
        )
        # aps_chunks.shape == num_bins * (B, H, N/num_bins), # idx_sorted.shape == num_bins * (B, H, N/num_bins)
        B, H, chunk_size = aps_chunks[0].shape
        assert H == 1, "Number of heads should be 1!"

        idx_batch_list = []
        k_batch_list = []
        for i in range(B):
            k_list = []
            idx_list = []
            for j in range(num_bins):
                # each bin has K samples
                if j != num_bins - 1:
                    k = int(2 * M / num_bins * bin_prob[i, j])
                else:
                    k = M - sum(k_list)
                k_list.append(k)

                if self.bin_sample_mode == "topk":
                    idx_tmp = aps_chunks[j][i].topk(k, dim=-1)[1]  # idx.shape == (H, k)
                elif self.bin_sample_mode == "uniform":
                    idx_tmp = torch.randperm(chunk_size)[:k]
                    idx_tmp = (
                        idx_tmp.unsqueeze(0)
                            .expand(H, -1)
                            .to(attention_point_score.device)
                    )
                elif self.bin_sample_mode == "random":
                    if k == 0:
                        continue
                    aps_chunks_tmp = ops.norm_range(
                        aps_chunks[j][i], dim=-1, n_min=0, n_max=1, mode="minmax"
                    )
                    aps_chunks_tmp = aps_chunks_tmp / (self.boltzmann_T + 1e-8)
                    aps_chunks_tmp = F.softmax(aps_chunks_tmp, dim=-1)
                    idx_tmp = torch.multinomial(
                        aps_chunks_tmp, num_samples=k, replacement=False
                    )
                else:
                    raise ValueError(
                        "Please check the setting of bin sample mode. It must be topk, multinomial or random!"
                    )
                idx = torch.gather(
                    idx_chunks[j][i], dim=-1, index=idx_tmp
                )  # idx.shape == (H, k)
                idx_list.append(idx)
            idx_single = torch.cat(idx_list, dim=-1)  # idx_list.shape == (H, M)
            idx_batch_list.append(idx_single)
            k_single = torch.tensor(k_list).to(attention_point_score.device)
            k_batch_list.append(k_single)
        idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)
        k_batch = torch.stack(k_batch_list, dim=0)  # k_batch.shape == (B, num_bins)
        return idx_batch, k_batch, idx_chunks

    # def nonuniform_bin_idx_selection(self, attention_point_score, bin_boundaries, bin_prob, normalization_mode):
    #     # bin_prob.shape == (B, num_bins)
    #     # self.attention_point_score.shape == (B, H, N)
    #     aps_chunks, idx_chunks = ops.sort_chunk_nonuniform(attention_point_score, bin_boundaries, normalization_mode)
    #     # print(f'idx.dtype3:{idx_chunks[0][0].dtype}')
    #     # aps_chunks.shape == num_bins * (B, H, n), # idx_sorted.shape == num_bins * (B, H, N/num_bins)
    #     num_bins = len(bin_boundaries) + 1
    #     B, H, N = attention_point_score.shape
    #
    #     # chunk_size = aps_chunks[j][i].shape[1]
    #     assert H == 1, "Number of heads should be 1!"
    #
    #     # k_point_to_choose = torch.zeros(B, num_bins).to(bin_prob.device)
    #     # for i in range(num_bins):
    #     #     for j in range(B):
    #     #         num_points_in_bin = aps_chunks[i][j].nelement()
    #     #         k_point_to_choose[j, i] = num_points_in_bin * bin_prob[j, i]
    #     # k_point_to_choose = k_point_to_choose * self.M / torch.sum(k_point_to_choose, dim=1, keepdim=True)
    #     # k_point_to_choose = k_point_to_choose.round().int()
    #
    #     # for _ in range(num_bins):
    #     #     k_point_to_drop = torch.empty_like(k_point_to_choose)
    #     #     for i in range(B):
    #     #         for j in range(num_bins):
    #     #             k_point_to_choose[i, j] = min(k_point_to_choose[i, j], aps_chunks[j][i].nelement())
    #     #             k_point_to_drop[i, j] = aps_chunks[j][i].nelement() - k_point_to_choose[i, j]
    #     #     correction_for_rouding = self.M - torch.sum(k_point_to_choose, dim=1)
    #     #
    #     #     if torch.max(torch.abs(correction_for_rouding)) == 0:
    #     #         break
    #     #     else:
    #     #         # assert torch.max(torch.abs(correction_for_rouding)) < 3, 'correction_for_rouding seems to be too big.'
    #     #         for i in range(B):
    #     #             k_point_to_choose[i, torch.argmax(k_point_to_drop[i, :])] += int(correction_for_rouding[i])
    #
    #     max_num_points = torch.zeros((B, num_bins), dtype=torch.long, device=bin_prob.device)
    #     for i in range(B):
    #         for j in range(num_bins):
    #             max_num_points[i, j] = aps_chunks[j][i].shape[1]
    #     # print(f'bin_prob{bin_prob}-----------')
    #     k_point_to_choose = calculate_num_points_to_choose(bin_prob, max_num_points, self.M)
    #     # print(f'k_point_to_choose{torch.sum(k_point_to_choose,dim=1)}')
    #
    #     idx_batch_list = []
    #     for i in range(B):
    #
    #         idx_list = []
    #         # print(f'sampling_scale:{sampling_scale}')
    #         # print(f'self.M:{self.M}')
    #         # print(f'chunk_size_list:{chunk_size_list}')
    #
    #         for j in range(num_bins):
    #             # each bin has k samples
    #             k = k_point_to_choose[i, j]
    #
    #             if self.bin_sample_mode == "topk":
    #                 idx_tmp = aps_chunks[j][i].topk(k, dim=-1)[1]  # idx.shape == (H, k)
    #             elif self.bin_sample_mode == "uniform":
    #                 idx_tmp = torch.randperm(aps_chunks[j][i].shape[1])[:k]
    #                 idx_tmp = idx_tmp.unsqueeze(0).expand(H, -1).to(attention_point_score.device)
    #             elif self.bin_sample_mode == "random":
    #                 if k != 0:
    #                     aps_chunks_tmp = ops.norm_range(aps_chunks[j][i], dim=-1, n_min=0, n_max=1, mode="minmax")
    #                     aps_chunks_tmp = aps_chunks_tmp / (self.boltzmann_T + 1e-8)
    #                     aps_chunks_tmp = F.softmax(aps_chunks_tmp, dim=-1)
    #                     # print(f'k:{k}')
    #                     # print(f'aps_chunks_tmp.shape:{aps_chunks_tmp.shape}')
    #                     if aps_chunks_tmp.nelement() < k:
    #                         print(f'aps_chunks_tmp{aps_chunks_tmp.nelement()},k{k}')
    #                         exit(-1)
    #                     idx_tmp = torch.multinomial(aps_chunks_tmp, num_samples=k, replacement=False)
    #             else:
    #                 raise ValueError(
    #                     'Please check the setting of bin sample mode. It must be topk, multinomial or random!')
    #             # print(f'k:{k}')
    #             # print(f'idx_tmp:{idx_tmp}')
    #             # print(f'idx_tmp.shape:{idx_tmp.shape}')
    #             # print(f'idx_chunks[j][i].shape:{idx_chunks[j][i].shape}')
    #             if k != 0:
    #                 idx = idx_chunks[j][i][0, idx_tmp[0]].reshape(1, -1)
    #                 # torch.gather(idx_chunks[j][i], dim=-1, index=idx_tmp)  # idx.shape == (H, k)
    #                 idx_list.append(idx)
    #         idx_single = torch.cat(idx_list, dim=-1)  # idx_list.shape == (H, M)
    #         idx_batch_list.append(idx_single)
    #     idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)
    #     # k_point_to_choose.shape == (B, num_bins)
    #     # print(f'idx.dtype2:{idx.dtype}')
    #     return idx_batch, k_point_to_choose, idx_chunks

    def bin2_idx_selection(self):
        # self.attention_point_score.shape == (B, H, N)
        aps_bins, idx_bins = ops.sort_chunk(
            self.attention_point_score, self.num_bins, dim=-1, descending=True
        )
        # aps_bins.shape == num_bins * (B, H, N/num_bins), # idx_sorted.shape == num_bins * (B, H, N/num_bins)
        B, H, bin_size = aps_bins[0].shape
        _, _, bin_size_min = aps_bins[-1].shape
        assert H == 1, "Number of heads should be 1!"
        assert (
                bin_size_min == bin_size
        ), "The number of points must be divisible by the number of bins!"
        aps_bins = torch.stack(
            aps_bins
        )  # aps_bins.shape == (num_bins, B, H, N/num_bins)
        aps_bins = aps_bins.permute(
            1, 2, 0, 3
        )  # aps_bins.shape == (B, H, num_bins, N/num_bins)
        aps_bins = torch.mean(aps_bins, dim=-1)  # aps_bins.shape == (B, H, num_bins)
        aps_bins = ops.norm_range(aps_bins, dim=-1, n_min=0, n_max=1, mode="minmax")
        aps_bins /= self.boltzmann_T + 1e-8
        aps_bins = F.softmax(aps_bins, dim=-1)  # aps_bins.shape == (B, H, num_bins)

        idx_batch_list = []
        k_batch_list = []
        for i in range(B):
            idx_bin = torch.multinomial(
                aps_bins[i, 0], num_samples=self.M, replacement=True
            )
            # adjust the number of samples in each bin
            count_list = []
            rest_count = 0
            for k in range(self.num_bins):
                count = torch.eq(idx_bin, k)
                count = torch.sum(count) + rest_count
                if k == self.num_bins - 1:
                    bin_size_tmp = bin_size_min
                else:
                    bin_size_tmp = bin_size
                if count > bin_size_tmp:
                    rest_count = count - bin_size_tmp
                    count = bin_size_tmp
                else:
                    rest_count = 0
                count_list.append(count)
            if rest_count > 0:
                for k, count in enumerate(count_list):
                    count += rest_count
                    if count > bin_size:
                        rest_count = count - bin_size
                        count_list[k] = bin_size
                    else:
                        rest_count = 0
                        break
            idx_list = []
            for k in range(self.num_bins):
                if k == self.num_bins - 1:
                    bin_size_tmp = bin_size_min
                else:
                    bin_size_tmp = bin_size
                idx_tmp = torch.randperm(bin_size_tmp)[: count_list[k]]
                idx_tmp = idx_tmp.unsqueeze(0).to(self.attention_point_score.device)
                idx = torch.gather(
                    idx_bins[k][i], dim=-1, index=idx_tmp
                )  # idx.shape == (H, k)
                idx_list.append(idx)
            idx_single = torch.cat(idx_list, dim=-1)  # idx_list.shape == (H, M)
            idx_batch_list.append(idx_single)
            k_single = torch.tensor(count_list).to(self.attention_point_score.device)
            k_batch_list.append(k_single)
        idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)
        k_batch = torch.stack(k_batch_list, dim=0)  # k_batch.shape == (B, num_bins)
        self.bin_prob = k_batch / self.M
        return idx_batch, k_batch

    def boltzmann_idx_selection(
            self, attention_point_score, M, boltzmann_norm_mode, boltzmann_T
    ):
        B, H, N = attention_point_score.shape
        aps_boltz = ops.norm_range(
            attention_point_score, dim=-1, n_min=0, n_max=1, mode=boltzmann_norm_mode
        )
        aps_boltz /= boltzmann_T
        aps_boltz = F.softmax(aps_boltz, dim=-1)
        idx_batch_list = []
        for i in range(B):
            idx_boltz_list = []
            for j in range(H):
                idx_boltz = torch.multinomial(
                    aps_boltz[i, j], num_samples=M, replacement=False
                )
                idx_boltz_list.append(idx_boltz)
            idx_single = torch.stack(idx_boltz_list, dim=0)
            idx_batch_list.append(idx_single)
        idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)
        return idx_batch


class _DownSampleToken(nn.Module):
    def __init__(self, config_ds, layer):
        super(DownSampleToken, self).__init__()

        self.M = config_ds.M[layer]
        self.K = 32
        self.asm = config_ds.asm[layer]
        self.res = config_ds.res.enable[layer]
        self.ff = config_ds.res.ff[layer]
        self.num_heads = config_ds.num_heads[layer]
        self.idx_mode = config_ds.idx_mode[layer]
        self.bin_mode = config_ds.bin.mode[layer]
        self.relu_mean_order = config_ds.bin.relu_mean_order[layer]

        self.num_bins = config_ds.bin.num_bins[layer]

        q_in = config_ds.q_in[layer]
        q_out = config_ds.q_out[layer]
        k_in = config_ds.k_in[layer]
        k_out = config_ds.k_out[layer]
        v_in = config_ds.v_in[layer]
        v_out = config_ds.v_out[layer]

        self.q_depth = int(q_out / self.num_heads)
        self.k_depth = int(k_out / self.num_heads)
        self.v_depth = int(v_out / self.num_heads)
        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)

        if self.bin_mode == "token":
            self.token_mode = config_ds.bin.token_mode[layer]

            if self.token_mode == "multi_token":
                self.bin_tokens = nn.Parameter(torch.randn(1, q_in, self.num_bins))
            elif self.token_mode == "one_token":
                self.bin_tokens = nn.Parameter(torch.randn(1, q_in, 1))

        else:
            raise NotImplementedError

        self.softmax = nn.Softmax(dim=-1)
        # downsample res link
        if self.res:
            self.bn1 = nn.BatchNorm1d(v_out)
            if self.ff:
                self.ffn = nn.Sequential(
                    nn.Conv1d(128, 512, 1, bias=False),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Conv1d(512, 128, 1, bias=False),
                )
                self.bn2 = nn.BatchNorm1d(v_out)
        # bin
        self.bin_enable = config_ds.bin.enable[layer]
        self.num_bins = config_ds.bin.num_bins[layer]
        self.scaling_factor = config_ds.bin.scaling_factor[layer]
        self.bin_sample_mode = config_ds.bin.sample_mode[layer]
        self.bin_norm_mode = config_ds.bin.norm_mode[layer]
        self.bin_mode = config_ds.bin.mode[layer]
        self.enable_multiply = config_ds.bin.multiply[layer]
        self.direct_link_mode = config_ds.bin.direct_link_mode[layer]
        self.momentum_update_factor = config_ds.bin.momentum_update_factor[layer]

        self.dynamic_boundaries = config_ds.bin.dynamic_boundaries
        if config_ds.bin.dynamic_boundaries:
            self.bin_boundaries = None
        else:
            # # self.bin_boundaries = config_ds.bin.bin_boundaries[layer]
            bin_boundaries_upper = [float("inf")]
            bin_boundaries_upper.extend(config_ds.bin.bin_boundaries[layer])
            bin_boundaries_lower = config_ds.bin.bin_boundaries[layer]
            bin_boundaries_lower.append(float("-inf"))
            self.bin_boundaries = [
                torch.asarray(bin_boundaries_upper).reshape(1, 1, 1, self.num_bins),
                # [inf, 0.503, 0.031, -0.230, -0.427, -0.627]
                torch.asarray(bin_boundaries_lower).reshape(1, 1, 1, self.num_bins),
                # [0.503, 0.031, -0.230, -0.427, -0.627, -inf]
            ]

        self.normalization_mode = config_ds.bin.normalization_mode[layer]

        # boltzmann
        self.boltzmann_enable = config_ds.boltzmann.enable[layer]
        self.boltzmann_T = config_ds.bin.boltzmann_T[layer]
        self.boltzmann_norm_mode = config_ds.boltzmann.norm_mode[layer]

    def forward(self, x, x_xyz=None):
        # x.shape == (B, C, N)

        B, C, N = x.shape
        if self.bin_mode == "token":
            bin_tokens = einops.repeat(
                self.bin_tokens, "1 c num_bins -> b c num_bins", b=B
            )
            # bin_tokens.shape ==(B,C,num_bins)
            x_and_token = torch.concat((x, bin_tokens), dim=2)  # x: (B,C,N+num_bins)

            q = self.q_conv(x)
            # q.shape == (B, C, N)
            q = self.split_heads(q, self.num_heads, self.q_depth)
            # q.shape == (B, H, D, N)
            k = self.k_conv(x_and_token)
            # k.shape ==  (B, C, N+num_bins)
            k = self.split_heads(k, self.num_heads, self.k_depth)
            # k.shape == (B, H, D, N+num_bins)
            v = self.v_conv(x_and_token)
            # v.shape ==  (B, C, N+num_bins)
            v = self.split_heads(v, self.num_heads, self.v_depth)
            # v.shape == (B, H, D, N+num_bins)
            q = q.permute(0, 1, 3, 2)  # q.shape == (B, H, N, D)

            attention_map, attention_map_beforesoftmax = self.attention_scoring(
                q, k
            )  # attention_map: (B,1,N,N+num_bins)

            attention_points, attention_bins = torch.split(attention_map, N, dim=-1)
            _, attention_bins_beforesoftmax = torch.split(
                attention_map_beforesoftmax, N, dim=-1
            )
            # attention_bins_beforesoftmax: (B,1,N,num_bins)

        else:
            raise NotImplementedError

        self.attention_point_score, self.sparse_attention_map, self.mask = (
            self.calculate_attention_score(x, attention_points)
        )

        if self.dynamic_boundaries:
            idx, k_point_to_choose, idx_chunks, self.bin_boundaries, bin_prob = (
                nonuniform_bin_idx_selection_beforesoftmaxbinprob(
                    self.attention_point_score,
                    self.bin_boundaries,
                    attention_bins_beforesoftmax,
                    self.normalization_mode,
                    self.M,
                    self.bin_sample_mode,
                    self.dynamic_boundaries,
                    self.relu_mean_order,
                    self.num_bins,
                    self.momentum_update_factor,
                    self.boltzmann_T,
                )
            )
            # bin_prob, _ = torch.max(attention_bins, dim=-2)  # x_bins: (B,1,num_bins)
            # bin_prob = bin_prob.squeeze(1)  # x_bins: (B,num_bins)
            # idx, k_point_to_choose, idx_chunks, self.bin_boundaries = nonuniform_bin_idx_selection(
            #     self.attention_point_score,
            #     self.bin_boundaries,
            #     bin_prob,
            #     self.normalization_mode,
            #     self.M,
            #     self.bin_sample_mode,
            #     self.dynamic_boundaries)
        else:
            bin_prob, _ = torch.max(attention_bins, dim=-2)  # x_bins: (B,1,num_bins)
            bin_prob = bin_prob.squeeze(1)  # x_bins: (B,num_bins)
            bin_boundaries = [item.to(x.device) for item in self.bin_boundaries]
            idx, k_point_to_choose, idx_chunks, _ = nonuniform_bin_idx_selection(
                self.attention_point_score,
                bin_boundaries,
                bin_prob,
                self.normalization_mode,
                self.M,
                self.bin_sample_mode,
                self.dynamic_boundaries,
            )
        # k_point_to_choose.shape == (B, num_bins)
        # idx_chunks.shape == num_bins * (B, H, n)

        attention_down = torch.gather(
            attention_map,
            dim=2,
            index=idx.unsqueeze(3).expand(-1, -1, -1, attention_map.shape[-1]),
        )
        # attention_down.shape == (B, H, M, N+num_bins)
        v_down = (attention_down @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # attention_down: (B, H, M, N+num_bins)
        # v.shape == (B, H, D, N+num_bins)
        # v_down.shape == (B, M, H, D)
        x_ds = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)

        # residual & feedforward
        if self.res is True:
            x_ds = self.res_block(x, x_ds, idx)

        if self.enable_multiply:
            x_ds = bin_probability_multiple(
                x_ds, x.shape, idx, idx_chunks, bin_prob, self.direct_link_mode
            )

        self.idx = idx
        self.idx_chunks = idx_chunks
        # idx_chunks.shape == num_bins * (B, H, n)
        self.bin_prob = bin_prob
        # bin_prob.shape == (B, num_bins)
        self.k_point_to_choose = k_point_to_choose
        # k_point_to_choose.shape == (B, num_bins)
        return (x_ds, idx), (None, None)

    def output_variables(self, *args):

        # print(vars().keys())
        variables = None
        for i, key in enumerate(args):
            if i == 0:
                variables = getattr(vars()["self"], key)
                # variables = vars()[f'self.{key}']
            elif i == 1:
                variables = (variables,) + (getattr(vars()["self"], key),)
                # variables = (variables,) + (vars()[f'self.{key}'],)
            else:
                variables = variables + (getattr(vars()["self"], key),)
                # variables = variables + (vars()[f'self.{key}'],)

        return variables

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x

    def attention_scoring(
            self, q, k
    ):  # q.shape == (B, H, N, D), k.shape == (B, H, D, N)
        if self.asm == "dot":
            energy = q @ k  # energy.shape == (B, H, N, N)
        elif self.asm == "l2":
            energy = -1 * ops.l2_global(q, k)  # -(Q-K)^2 energy.shape == (B, H, N, N)
        elif self.asm == "l2+":
            energy = ops.l2_global(q, k)  # (Q-K)^2 energy.shape == (B, H, N, N)
        else:
            raise ValueError("Please check the setting of asm!")

        scale_factor = math.sqrt(q.shape[-1])

        attention = self.softmax(
            energy / scale_factor
        )  # attention.shape == (B, H, N, N)

        return attention, energy / scale_factor

    def res_block(self, x, x_ds, idx):  # x.shape == (B, C, N), x_ds.shape == (B, C, M)
        x_tmp = torch.gather(x, dim=-1, index=idx)  # x_res.shape == (B, 1, M)
        x_res = self.bn1(x_ds + x_tmp)  # x_res.shape == (B, C, M)
        if self.ff == True:
            x_tmp = self.ffn(x_res)
            x_res = self.bn2(x_ds + x_tmp)
        return x_res  # x_res.shape == (B, C, M)

    def get_sparse_attention_map(self, x, attention_points):
        mask = ops.neighbor_mask(x, self.K)
        mask = mask.unsqueeze(1).expand(-1, attention_points.shape[1], -1, -1)
        # print(f'attention_map.shape{self.attention_map.shape}')
        # print(f'mask.shape{mask.shape}')
        # exit(-1)
        sparse_attention_map = attention_points * mask
        return mask, sparse_attention_map

    def calculate_attention_score(self, x, attention_points):
        mask, sparse_attention_map = self.get_sparse_attention_map(x, attention_points)
        sparse_num = torch.sum(mask, dim=-2) + 1e-8

        # full attention map based
        if self.idx_mode == "col_sum":
            attention_point_score = torch.sum(
                attention_points, dim=-2
            )  # self.attention_point_score.shape == (B, H, N)
        elif self.idx_mode == "row_std":
            attention_point_score = torch.std(attention_points, dim=-1)

        # sparse attention map based

        elif self.idx_mode == "sparse_row_sum":
            attention_point_score = torch.sum(sparse_attention_map, dim=-1)
        elif self.idx_mode == "sparse_row_std":
            sparse_attention_map_std = sparse_attention_map.masked_select(
                mask != 0
            ).view(sparse_attention_map.shape[:-1] + (self.K,))
            attention_point_score = torch.std(sparse_attention_map_std, dim=-1)
        elif self.idx_mode == "sparse_col_sum":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2)
        elif self.idx_mode == "sparse_col_avg":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2) / sparse_num
        elif self.idx_mode == "sparse_col_sqr":
            attention_point_score = (
                    torch.sum(sparse_attention_map, dim=-2) / sparse_num / sparse_num
            )
        else:
            raise ValueError("Please check the setting of idx mode!")

        return attention_point_score, sparse_attention_map, mask


class DownSampleToken(nn.Module):
    def __init__(self, config_ds, layer):
        super(DownSampleToken, self).__init__()

        self.M = config_ds.M[layer]
        self.K = config_ds.K
        self.asm = config_ds.asm[layer]
        self.res = config_ds.res.enable[layer]
        self.ff = config_ds.res.ff[layer]
        self.num_heads = config_ds.num_heads[layer]
        self.idx_mode = config_ds.idx_mode[layer]
        self.bin_mode = config_ds.bin.mode[layer]
        self.relu_mean_order = config_ds.bin.relu_mean_order[layer]

        self.num_bins = config_ds.bin.num_bins[layer]

        q_in = config_ds.q_in[layer]
        q_out = config_ds.q_out[layer]
        k_in = config_ds.k_in[layer]
        k_out = config_ds.k_out[layer]
        v_in = config_ds.v_in[layer]
        v_out = config_ds.v_out[layer]

        self.q_depth = int(q_out / self.num_heads)
        self.k_depth = int(k_out / self.num_heads)
        self.v_depth = int(v_out / self.num_heads)
        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)

        if self.bin_mode == "token":
            self.token_mode = config_ds.bin.token_mode[layer]

            if self.token_mode == "multi_token":
                # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, self.num_bins))
                # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, self.num_bins)/torch.sqrt(q_in))
                self.bin_tokens = nn.Parameter(
                    torch.normal(
                        mean=0, std=1 / math.sqrt(q_in), size=(1, q_in, self.num_bins)
                    )
                )
            elif self.token_mode == "one_token":
                # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, 1))
                # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, 1))
                self.bin_tokens = nn.Parameter(
                    torch.normal(mean=0, std=1 / math.sqrt(q_in), size=(1, q_in, 1))
                )

        else:
            raise NotImplementedError

        self.softmax = nn.Softmax(dim=-1)
        # downsample res link
        if self.res:
            self.bn1 = nn.BatchNorm1d(v_out)
            if self.ff:
                self.ffn = nn.Sequential(
                    nn.Conv1d(128, 512, 1, bias=False),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Conv1d(512, 128, 1, bias=False),
                )
                self.bn2 = nn.BatchNorm1d(v_out)
        # bin
        self.bin_enable = config_ds.bin.enable[layer]
        self.num_bins = config_ds.bin.num_bins[layer]
        self.scaling_factor = config_ds.bin.scaling_factor[layer]
        self.bin_sample_mode = config_ds.bin.sample_mode[layer]
        self.bin_norm_mode = config_ds.bin.norm_mode[layer]
        self.bin_mode = config_ds.bin.mode[layer]
        self.enable_multiply = config_ds.bin.multiply[layer]
        self.direct_link_mode = config_ds.bin.direct_link_mode[layer]
        self.momentum_update_factor = config_ds.bin.momentum_update_factor[layer]

        self.dynamic_boundaries = config_ds.bin.dynamic_boundaries
        if config_ds.bin.dynamic_boundaries:
            self.bin_boundaries = None
        else:
            # # self.bin_boundaries = config_ds.bin.bin_boundaries[layer]
            bin_boundaries_upper = [float("inf")]
            bin_boundaries_upper.extend(config_ds.bin.bin_boundaries[layer])
            bin_boundaries_lower = config_ds.bin.bin_boundaries[layer]
            bin_boundaries_lower.append(float("-inf"))
            self.bin_boundaries = [
                torch.asarray(bin_boundaries_upper).reshape(1, 1, 1, self.num_bins),
                # [inf, 0.503, 0.031, -0.230, -0.427, -0.627]
                torch.asarray(bin_boundaries_lower).reshape(1, 1, 1, self.num_bins),
                # [0.503, 0.031, -0.230, -0.427, -0.627, -inf]
            ]

        self.normalization_mode = config_ds.bin.normalization_mode[layer]

        # boltzmann
        self.boltzmann_enable = config_ds.boltzmann.enable[layer]
        self.boltzmann_T = config_ds.bin.boltzmann_T[layer]
        self.boltzmann_norm_mode = config_ds.boltzmann.norm_mode[layer]

        self.token_orthognonal_loss_factor = config_ds.bin.token_orthognonal_loss_factor

    def forward(self, x, x_xyz=None):
        # x.shape == (B, C, N)

        B, C, N = x.shape
        if self.bin_mode == "token":
            bin_tokens = einops.repeat(
                self.bin_tokens, "1 c num_bins -> b c num_bins", b=B
            )
            # bin_tokens.shape ==(B,C,num_bins)
            x_and_token = torch.concat((x, bin_tokens), dim=2)  # x: (B,C,N+num_bins)

            if self.asm == "dot":

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

                attention_map = self.softmax(
                    attention_map_beforesoftmax
                )  # attention.shape == (B, H, N, N+num_bins)

                _, attention_bins_beforesoftmax = torch.split(
                    attention_map_beforesoftmax, N, dim=-1
                )
                # attention_bins_beforesoftmax: (B,1,N,num_bins)
                attention_points, attention_bins = torch.split(attention_map, N, dim=-1)
            elif self.asm == "l2":
                q = self.q_conv(x_and_token)
                # q.shape == (B, C, N+num_bins)
                q = self.split_heads(q, self.num_heads, self.q_depth)
                q = q.permute(0, 1, 3, 2)  # q.shape == (B, H, N+num_bins, D)

                # q.shape == (B, H, D, N)
                k = self.k_conv(x_and_token)
                # k.shape ==  (B, C, N+num_bins)
                k = self.split_heads(k, self.num_heads, self.k_depth)
                # k.shape == (B, H, D, N+num_bins)
                v = self.v_conv(x_and_token)
                # v.shape ==  (B, C, N+num_bins)
                v = self.split_heads(v, self.num_heads, self.v_depth)
                # v.shape == (B, H, D, N+num_bins)

                energy = -1 * ops.l2_global(
                    q, k
                )  # -(Q-K)^2 energy.shape == (B, H, N+num_bins, N+num_bins)

                scale_factor = math.sqrt(q.shape[-1])

                attention_map_beforesoftmax = energy / scale_factor
                # attention_map_beforesoftmax.shape == (B, H, N+num_bins, N+num_bins)
                attention_map_beforesoftmax, _ = torch.split(
                    attention_map_beforesoftmax, N, dim=2
                )
                # attention_map_beforesoftmax.shape == (B, H, N, N+num_bins)

                attention_map = self.softmax(
                    attention_map_beforesoftmax
                )  # attention.shape == (B, H, N, N+num_bins)

                _, attention_bins_beforesoftmax = torch.split(
                    attention_map_beforesoftmax, N, dim=-1
                )
                # attention_bins_beforesoftmax: (B,1,N,num_bins)
                attention_points, attention_bins = torch.split(attention_map, N, dim=-1)

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.attention_point_score, _, _ = self.calculate_attention_score(
            x, attention_points
        )
        # self.attention_point_score: (B, H, N)

        self.bin_boundaries, self.bin_points_mask = bin_partition(
            self.attention_point_score,
            self.bin_boundaries,
            self.dynamic_boundaries,
            self.momentum_update_factor,
            self.normalization_mode,
            self.num_bins,
        )
        # self.bin_points_mask: (B,H,N,num_bins)
        # normalized_attention_point_score: (B,H,N)

        bin_weights, self.bin_weights_beforerelu = self.bin_weghts_calculation(
            attention_bins_beforesoftmax, self.bin_points_mask, self.relu_mean_order
        )

        # self.bin_points_mask: (B, H, N, num_bins)
        max_num_points = torch.sum(self.bin_points_mask.squeeze(dim=1), dim=1)
        # max_num_points:(B,num_bins)
        self.k_point_to_choose = calculate_num_points_to_choose(
            bin_weights, max_num_points, self.M
        )
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
            self.M,
            self.attention_point_score,
            self.bin_points_mask,
            self.bin_sample_mode,
            self.boltzmann_T,
            self.k_point_to_choose,
        )

        attention_down = torch.gather(
            attention_map,
            dim=2,
            index=index_down.unsqueeze(3).expand(-1, -1, -1, attention_map.shape[-1]),
        )
        # attention_down.shape == (B, H, M, N+num_bins)
        v_down = (attention_down @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # attention_down: (B, H, M, N+num_bins)
        # v.shape == (B, H, D, N+num_bins)
        # v_down.shape == (B, M, H, D)
        x_ds = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)

        # residual & feedforward
        if self.res is True:
            x_ds = self.res_block(x, x_ds, index_down)

        self.idx = index_down
        # self.idx: (B,H,M)

        self.attention_bins_beforesoftmax = attention_bins_beforesoftmax
        return (x_ds, index_down), (None, None)

    def bin_weghts_calculation(
            self, attention_bins_beforesoftmax, bin_points_mask, relu_mean_order
    ):
        masked_attention_map_token = attention_bins_beforesoftmax * bin_points_mask
        if relu_mean_order == "mean_relu":

            bin_weights_beforerelu = torch.sum(masked_attention_map_token, dim=2) / (
                    torch.count_nonzero(bin_points_mask, dim=2) + 1e-8
            )
            # torch.count_nonzero(masked_attention_map_token, dim=2) + 1e-8)
            bin_weights_beforerelu = bin_weights_beforerelu.squeeze(1)
            bin_weights = F.relu(bin_weights_beforerelu)
        elif relu_mean_order == "relu_mean":
            masked_attention_map_token = F.relu(masked_attention_map_token)
            bin_weights_beforerelu = torch.sum(masked_attention_map_token, dim=2) / (
                    torch.count_nonzero(bin_points_mask, dim=2) + 1e-8
            )
            bin_weights_beforerelu = bin_weights_beforerelu.squeeze(1)
            bin_weights = bin_weights_beforerelu
        else:
            raise NotImplementedError
        return bin_weights, bin_weights_beforerelu

    def output_variable_calculatio(self):
        # 'idx_chunks'
        B, _, _, num_bins = self.bin_points_mask.shape

        index_batch, _, index_point, index_bin = torch.where(self.bin_points_mask)

        self.idx_chunks = [
            [
                index_point[(index_bin == i) & (index_batch == j)].reshape(1, -1)
                for j in range(B)
            ]
            for i in range(num_bins)
        ]

        # 'bin_prob'
        self.bin_prob = self.bin_weights_beforerelu
        # bin_prob.shape == (B, num_bins)

    def output_variables(self, *args):

        # print(vars().keys())
        variables = None
        for i, key in enumerate(args):
            if i == 0:
                variables = getattr(vars()["self"], key)
                # variables = vars()[f'self.{key}']
            elif i == 1:
                variables = (variables,) + (getattr(vars()["self"], key),)
                # variables = (variables,) + (vars()[f'self.{key}'],)
            else:
                variables = variables + (getattr(vars()["self"], key),)
                # variables = variables + (vars()[f'self.{key}'],)

        return variables

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
        mask = ops.neighbor_mask(x, self.K)
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
        if self.idx_mode == "col_sum":
            attention_point_score = torch.sum(
                attention_points, dim=-2
            )  # self.attention_point_score.shape == (B, H, N)
        elif self.idx_mode == "row_std":
            attention_point_score = torch.std(attention_points, dim=-1)

        # sparse attention map based

        elif self.idx_mode == "sparse_row_sum":
            attention_point_score = torch.sum(sparse_attention_map, dim=-1)
        elif self.idx_mode == "sparse_row_std":
            sparse_attention_map_std = sparse_attention_map.masked_select(
                mask != 0
            ).view(sparse_attention_map.shape[:-1] + (self.K,))
            attention_point_score = torch.std(sparse_attention_map_std, dim=-1)
        elif self.idx_mode == "sparse_col_sum":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2)
        elif self.idx_mode == "sparse_col_avg":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2) / sparse_num
        elif self.idx_mode == "sparse_col_sqr":
            attention_point_score = (
                    torch.sum(sparse_attention_map, dim=-2) / sparse_num / sparse_num
            )
        else:
            raise ValueError("Please check the setting of idx mode!")

        attention_point_score[torch.isnan(attention_point_score)] = 0

        return attention_point_score, sparse_attention_map, mask


class DownSampleReparameterization(nn.Module):
    def __init__(self, config_ds, layer):
        super().__init__()

        self.M = config_ds.M[layer]
        self.K = config_ds.K
        self.asm = config_ds.asm[layer]
        self.res = config_ds.res.enable[layer]
        self.ff = config_ds.res.ff[layer]
        self.num_heads = config_ds.num_heads[layer]
        self.idx_mode = config_ds.idx_mode[layer]
        self.bin_mode = config_ds.bin.mode[layer]
        self.relu_mean_order = config_ds.bin.relu_mean_order[layer]

        self.num_bins = config_ds.bin.num_bins[layer]

        q_in = config_ds.q_in[layer]
        q_out = config_ds.q_out[layer]
        k_in = config_ds.k_in[layer]
        k_out = config_ds.k_out[layer]
        v_in = config_ds.v_in[layer]
        v_out = config_ds.v_out[layer]

        self.q_depth = int(q_out / self.num_heads)
        self.k_depth = int(k_out / self.num_heads)
        self.v_depth = int(v_out / self.num_heads)
        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)

        if self.bin_mode == "token":
            self.token_mode = config_ds.bin.token_mode[layer]

            if self.token_mode == "multi_token":
                # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, self.num_bins))
                # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, self.num_bins)/torch.sqrt(q_in))
                self.bin_tokens = nn.Parameter(
                    torch.normal(
                        mean=0, std=1 / math.sqrt(q_in), size=(1, q_in, self.num_bins)
                    )
                )
            elif self.token_mode == "one_token":
                # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, 1))
                # self.bin_tokens = nn.Parameter(torch.randn(1, q_in, 1))
                self.bin_tokens = nn.Parameter(
                    torch.normal(mean=0, std=1 / math.sqrt(q_in), size=(1, q_in, 1))
                )

        else:
            raise NotImplementedError

        self.softmax = nn.Softmax(dim=-1)
        # downsample res link
        if self.res:
            self.bn1 = nn.BatchNorm1d(v_out)
            if self.ff:
                self.ffn = nn.Sequential(
                    nn.Conv1d(128, 512, 1, bias=False),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Conv1d(512, 128, 1, bias=False),
                )
                self.bn2 = nn.BatchNorm1d(v_out)
        # bin
        self.bin_enable = config_ds.bin.enable[layer]
        self.num_bins = config_ds.bin.num_bins[layer]
        self.scaling_factor = config_ds.bin.scaling_factor[layer]
        self.bin_sample_mode = config_ds.bin.sample_mode[layer]
        self.bin_norm_mode = config_ds.bin.norm_mode[layer]
        self.bin_mode = config_ds.bin.mode[layer]
        self.enable_multiply = config_ds.bin.multiply[layer]
        self.direct_link_mode = config_ds.bin.direct_link_mode[layer]
        self.momentum_update_factor = config_ds.bin.momentum_update_factor[layer]

        self.dynamic_boundaries = config_ds.bin.dynamic_boundaries
        if config_ds.bin.dynamic_boundaries:
            self.bin_boundaries = None
        else:
            # # self.bin_boundaries = config_ds.bin.bin_boundaries[layer]
            bin_boundaries_upper = [float("inf")]
            bin_boundaries_upper.extend(config_ds.bin.bin_boundaries[layer])
            bin_boundaries_lower = config_ds.bin.bin_boundaries[layer]
            bin_boundaries_lower.append(float("-inf"))
            self.bin_boundaries = [
                torch.asarray(bin_boundaries_upper).reshape(1, 1, 1, self.num_bins),
                # [inf, 0.503, 0.031, -0.230, -0.427, -0.627]
                torch.asarray(bin_boundaries_lower).reshape(1, 1, 1, self.num_bins),
                # [0.503, 0.031, -0.230, -0.427, -0.627, -inf]
            ]

        self.normalization_mode = config_ds.bin.normalization_mode[layer]

        # boltzmann
        self.boltzmann_enable = config_ds.boltzmann.enable[layer]
        self.boltzmann_T = config_ds.bin.boltzmann_T[layer]
        self.boltzmann_norm_mode = config_ds.boltzmann.norm_mode[layer]

        self.token_orthognonal_loss_factor = config_ds.bin.token_orthognonal_loss_factor

    def forward(self, x, x_xyz=None):
        # x.shape == (B, C, N)

        B, C, N = x.shape
        if self.bin_mode == "token":
            bin_tokens = einops.repeat(
                self.bin_tokens, "1 c num_bins -> b c num_bins", b=B
            )
            # bin_tokens.shape ==(B,C,num_bins)
            x_and_token = torch.concat((x, bin_tokens), dim=2)  # x: (B,C,N+num_bins)

            if self.asm == "dot":

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

                attention_map = self.softmax(
                    attention_map_beforesoftmax
                )  # attention.shape == (B, H, N, N+num_bins)

                _, attention_bins_beforesoftmax = torch.split(
                    attention_map_beforesoftmax, N, dim=-1
                )
                # attention_bins_beforesoftmax: (B,1,N,num_bins)
                attention_points, attention_bins = torch.split(attention_map, N, dim=-1)
            elif self.asm == "l2":
                q = self.q_conv(x_and_token)
                # q.shape == (B, C, N+num_bins)
                q = self.split_heads(q, self.num_heads, self.q_depth)
                q = q.permute(0, 1, 3, 2)  # q.shape == (B, H, N+num_bins, D)

                # q.shape == (B, H, D, N)
                k = self.k_conv(x_and_token)
                # k.shape ==  (B, C, N+num_bins)
                k = self.split_heads(k, self.num_heads, self.k_depth)
                # k.shape == (B, H, D, N+num_bins)
                v = self.v_conv(x_and_token)
                # v.shape ==  (B, C, N+num_bins)
                v = self.split_heads(v, self.num_heads, self.v_depth)
                # v.shape == (B, H, D, N+num_bins)

                energy = -1 * ops.l2_global(
                    q, k
                )  # -(Q-K)^2 energy.shape == (B, H, N+num_bins, N+num_bins)

                scale_factor = math.sqrt(q.shape[-1])

                attention_map_beforesoftmax = energy / scale_factor
                # attention_map_beforesoftmax.shape == (B, H, N+num_bins, N+num_bins)
                attention_map_beforesoftmax, _ = torch.split(
                    attention_map_beforesoftmax, N, dim=2
                )
                # attention_map_beforesoftmax.shape == (B, H, N, N+num_bins)

                attention_map = self.softmax(
                    attention_map_beforesoftmax
                )  # attention.shape == (B, H, N, N+num_bins)

                _, attention_bins_beforesoftmax = torch.split(
                    attention_map_beforesoftmax, N, dim=-1
                )
                # attention_bins_beforesoftmax: (B,1,N,num_bins)
                attention_points, attention_bins = torch.split(attention_map, N, dim=-1)

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.attention_point_score, _, _ = self.calculate_attention_score(
            x, attention_points
        )
        # self.attention_point_score: (B, H, N)

        self.bin_boundaries, self.bin_points_mask = bin_partition(
            self.attention_point_score,
            self.bin_boundaries,
            self.dynamic_boundaries,
            self.momentum_update_factor,
            self.normalization_mode,
            self.num_bins,
        )
        # self.bin_points_mask: (B,H,N,num_bins)
        # normalized_attention_point_score: (B,H,N)

        bin_weights, self.bin_weights_beforerelu = self.bin_weghts_calculation(
            attention_bins_beforesoftmax, self.bin_points_mask, self.relu_mean_order
        )

        # self.bin_points_mask: (B, H, N, num_bins)
        max_num_points = torch.sum(self.bin_points_mask.squeeze(dim=1), dim=1)
        # max_num_points:(B,num_bins)
        self.k_point_to_choose = calculate_num_points_to_choose(
            bin_weights, max_num_points, self.M
        )
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
            self.M,
            self.attention_point_score,
            self.bin_points_mask,
            self.bin_sample_mode,
            self.boltzmann_T,
            self.k_point_to_choose,
        )

        attention_down = torch.gather(
            attention_map,
            dim=2,
            index=index_down.unsqueeze(3).expand(-1, -1, -1, attention_map.shape[-1]),
        )
        # attention_down.shape == (B, H, M, N+num_bins)
        v_down = (attention_down @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # attention_down: (B, H, M, N+num_bins)
        # v.shape == (B, H, D, N+num_bins)
        # v_down.shape == (B, M, H, D)
        x_ds = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)

        # residual & feedforward
        if self.res is True:
            x_ds = self.res_block(x, x_ds, index_down)

        self.idx = index_down
        # self.idx: (B,H,M)

        self.attention_bins_beforesoftmax = attention_bins_beforesoftmax
        return (x_ds, index_down), (None, None)

    def bin_weghts_calculation(
            self, attention_bins_beforesoftmax, bin_points_mask, relu_mean_order
    ):
        masked_attention_map_token = attention_bins_beforesoftmax * bin_points_mask
        if relu_mean_order == "mean_relu":

            bin_weights_beforerelu = torch.sum(masked_attention_map_token, dim=2) / (
                    torch.count_nonzero(bin_points_mask, dim=2) + 1e-8
            )
            # torch.count_nonzero(masked_attention_map_token, dim=2) + 1e-8)
            bin_weights_beforerelu = bin_weights_beforerelu.squeeze(1)
            bin_weights = F.relu(bin_weights_beforerelu)
        elif relu_mean_order == "relu_mean":
            masked_attention_map_token = F.relu(masked_attention_map_token)
            bin_weights_beforerelu = torch.sum(masked_attention_map_token, dim=2) / (
                    torch.count_nonzero(bin_points_mask, dim=2) + 1e-8
            )
            bin_weights_beforerelu = bin_weights_beforerelu.squeeze(1)
            bin_weights = bin_weights_beforerelu
        else:
            raise NotImplementedError
        return bin_weights, bin_weights_beforerelu

    def output_variable_calculatio(self):
        # 'idx_chunks'
        B, _, _, num_bins = self.bin_points_mask.shape

        index_batch, _, index_point, index_bin = torch.where(self.bin_points_mask)

        self.idx_chunks = [
            [
                index_point[(index_bin == i) & (index_batch == j)].reshape(1, -1)
                for j in range(B)
            ]
            for i in range(num_bins)
        ]

        # 'bin_prob'
        self.bin_prob = self.bin_weights_beforerelu
        # bin_prob.shape == (B, num_bins)

    def output_variables(self, *args):

        # print(vars().keys())
        variables = None
        for i, key in enumerate(args):
            if i == 0:
                variables = getattr(vars()["self"], key)
                # variables = vars()[f'self.{key}']
            elif i == 1:
                variables = (variables,) + (getattr(vars()["self"], key),)
                # variables = (variables,) + (vars()[f'self.{key}'],)
            else:
                variables = variables + (getattr(vars()["self"], key),)
                # variables = variables + (vars()[f'self.{key}'],)

        return variables

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
        mask = ops.neighbor_mask(x, self.K)
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
        if self.idx_mode == "col_sum":
            attention_point_score = torch.sum(
                attention_points, dim=-2
            )  # self.attention_point_score.shape == (B, H, N)
        elif self.idx_mode == "row_std":
            attention_point_score = torch.std(attention_points, dim=-1)

        # sparse attention map based

        elif self.idx_mode == "sparse_row_sum":
            attention_point_score = torch.sum(sparse_attention_map, dim=-1)
        elif self.idx_mode == "sparse_row_std":
            sparse_attention_map_std = sparse_attention_map.masked_select(
                mask != 0
            ).view(sparse_attention_map.shape[:-1] + (self.K,))
            attention_point_score = torch.std(sparse_attention_map_std, dim=-1)
        elif self.idx_mode == "sparse_col_sum":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2)
        elif self.idx_mode == "sparse_col_avg":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2) / sparse_num
        elif self.idx_mode == "sparse_col_sqr":
            attention_point_score = (
                    torch.sum(sparse_attention_map, dim=-2) / sparse_num / sparse_num
            )
        else:
            raise ValueError("Please check the setting of idx mode!")

        attention_point_score[torch.isnan(attention_point_score)] = 0

        return attention_point_score, sparse_attention_map, mask


class DownSampleInsert(nn.Module):
    def __init__(self, config_ds, layer):
        super(DownSampleInsert, self).__init__()
        self.M = config_ds.M[layer]
        self.K = 32
        self.asm = config_ds.asm[layer]
        self.res = config_ds.res.enable[layer]
        self.ff = config_ds.res.ff[layer]
        self.num_heads = config_ds.num_heads[layer]
        self.idx_mode = config_ds.idx_mode[layer]
        q_in = config_ds.q_in[layer]
        q_out = config_ds.q_out[layer]
        k_in = config_ds.k_in[layer]
        k_out = config_ds.k_out[layer]
        v_in = config_ds.v_in[layer]
        v_out = config_ds.v_out[layer]

        if self.asm == "dot":
            self.group_type = 'diff'
        else:
            self.group_type = "neighbor"

        self.q_depth = int(q_out / self.num_heads)
        self.k_depth = int(k_out / self.num_heads)
        self.v_depth = int(v_out / self.num_heads)

        self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # downsample res link
        if self.res:
            self.bn1 = nn.BatchNorm1d(v_out)
            if self.ff:
                self.ffn = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False),
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv1d(512, 128, 1, bias=False))
                self.bn2 = nn.BatchNorm1d(v_out)

        # bin
        self.bin_enable = config_ds.bin.enable[layer]
        self.bin_mode = config_ds.bin.mode[layer]
        self.num_bins = config_ds.bin.num_bins[layer]
        self.scaling_factor = config_ds.bin.scaling_factor[layer]
        self.bin_sample_mode = config_ds.bin.sample_mode[layer]
        self.bin_norm_mode = config_ds.bin.norm_mode[layer]
        if self.bin_enable and self.bin_mode == "mode1":
            self.bin_conv1 = nn.Conv1d(q_in, int(self.num_bins / 2), 1, bias=False)
            self.bin_conv2 = nn.Conv1d(q_in + int(self.num_bins / 2), q_out, 1, bias=False)
        # boltzmann
        self.boltzmann_enable = config_ds.boltzmann.enable[layer]
        self.boltzmann_T = config_ds.boltzmann.boltzmann_T[layer]
        self.boltzmann_norm_mode = config_ds.boltzmann.norm_mode[layer]

    def forward(self, x, x_xyz=None):
        # x.shape == (B, C, N)
        if self.bin_enable and self.bin_mode == "mode1":
            x, self.bin_prob = self.bin_conv(x)
        neighbors, self.neighbors_idx = ops.group(x, self.K, self.group_type)
        # neighbors.shape == (B, C, N, K)
        q = self.q_conv(x[:, :, :, None])
        # q.shape == (B, C, N, 1)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, N, 1, D)
        k = self.k_conv(neighbors)
        # k.shape ==  (B, C, N, K)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, N, K, D)
        v = self.v_conv(neighbors)
        # v.shape ==  (B, C, N, K)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, N, K, D)
        k = k.permute(0, 1, 2, 4, 3)
        # k.shape == (B, H, N, D, K)

        self.attention_map = self.attention_scoring(q, k)  # attention.shape == (B, H, N, 1, K)
        # sparse_attention_map = self.get_sparse_attention_map(neighbors_idx)
        self.idx, self.attention_point_score, self.sparse_attention_map, self.mask = self.idx_selection()
        if self.bin_enable:
            if self.bin_mode == "mode1":
                self.idx, self.bin_k = self.bin_idx_selection()
            elif self.bin_mode == "mode2":
                self.idx, self.bin_k = self.bin2_idx_selection()
            else:
                raise NotImplementedError
        elif self.boltzmann_enable:
            self.idx = self.boltzmann_idx_selection()
        idx_dropped = \
            torch.std(self.attention_map, dim=-1, unbiased=False)[:, :, :, 0].topk(
                self.attention_map.shape[-3] - self.M,
                dim=-1, largest=False)[1]
        # idx_dropped.shape == (B, H, N-M)
        attention_down = torch.gather(self.attention_map, dim=2,
                                      index=self.idx[..., None, None].expand(-1, -1, -1, -1, k.shape[-1]))
        # attention_down.shape == (B, H, M, 1, K)
        attention_dropped = torch.gather(self.attention_map, dim=2,
                                         index=idx_dropped[..., None, None].expand(-1, -1, -1, -1, k.shape[-1]))
        # attention_dropped.shape == (B, H, N-M, 1, K)
        v_down = torch.gather(v, dim=2, index=self.idx[..., None, None].expand(-1, -1, -1, k.shape[-1], k.shape[-2]))
        # v_down.shape == (B, H, M, K, D)
        v_dropped = torch.gather(v, dim=2,
                                 index=idx_dropped[..., None, None].expand(-1, -1, -1, k.shape[-1], k.shape[-2]))
        # v_dropped.shape == (B, H, N-M, K, D)
        v_down = (attention_down @ v_down)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # v_down.shape == (B, M, H, D)
        v_dropped = (attention_dropped @ v_dropped)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # v_dropped.shape == (B, N-M, H, D)
        x_ds = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)
        # x_ds.shape == (B, C, M)

        # residual & feedforward
        if self.res == True:
            x_ds = self.res_block(x, x_ds)

        x_dropped = v_dropped.reshape(v_dropped.shape[0], v_dropped.shape[1], -1).permute(0, 2, 1)
        # v_dropped.shape == (B, C, N-M)
        return (x_ds, self.idx), (x_dropped, idx_dropped)

        # 'bin_prob'
        # self.bin_prob = self.bin_weights_beforerelu
        # # bin_prob.shape == (B, num_bins)

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x

    def attention_scoring(self, q, k):  # q.shape == B, H, N, 1, D), k.shape == (B, H, N, D, K)
        if self.asm == "dot" or self.asm == "dot-neighbor":
            energy = q @ k  # energy.shape == (B, H, N, 1, K)
        elif self.asm == "dot-sub":
            energy = q @ (q.transpose(-1, -2) - k)  # Q@(Q-K)
        elif self.asm == "l2":
            energy = -1 * (q - k.transpose(-1, -2)) @ (
                    q.transpose(-1, -2) - k)  # -(Q-K)^2 # energy.shape == (B, H, N, K, K)
            energy = torch.mean(energy, dim=-2)  # energy.shape == (B, H, N, K)
            energy = energy.unsqueeze(-2)  # energy.shape == (B, H, N, 1, K)
        elif self.asm == "l2+":
            energy = (q - k.transpose(-1, -2)) @ (q.transpose(-1, -2) - k)  # (Q-K)^2 energy.shape == (B, H, N, K, K)
            energy = torch.mean(energy, dim=-2)  # energy.shape == (B, H, N, K)
            energy = energy.unsqueeze(-2)  # energy.shape == (B, H, N, 1, K)
        else:
            raise ValueError('Please check the setting of asm!')
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # attention.shape == (B, H, N, 1, K)
        return attention

    def res_block(self, x, x_ds):  # x.shape == (B, C, N), x_ds.shape == (B, C, M)
        x_tmp = torch.gather(x, dim=-1, index=self.idx)  # x_res.shape == (B, 1, M)
        x_res = self.bn1(x_ds + x_tmp)  # x_res.shape == (B, C, M)
        if self.ff == True:
            x_tmp = self.ffn(x_res)
            x_res = self.bn2(x_ds + x_tmp)
        return x_res  # x_res.shape == (B, C, M)

    def get_sparse_attention_map(self):
        attention_map = self.attention_map.squeeze(-2)
        B, H, N, K = attention_map.shape
        idx = self.neighbors_idx.view(B, H, N, K)
        sparse_attention_map = torch.zeros(B, H, N, N, dtype=torch.float32, device=idx.device).scatter_(-1, idx,
                                                                                                        attention_map)
        mask = torch.zeros(B, H, N, N, dtype=torch.float32, device=idx.device).scatter_(-1, idx, 1.0)
        return mask, sparse_attention_map

    def idx_selection(self):
        mask, sparse_attention_map = self.get_sparse_attention_map()
        sparse_num = torch.sum(mask, dim=-2) + 1e-8

        if self.idx_mode == "local_std":
            attention_point_score = torch.std(self.attention_map, dim=-1, unbiased=False)[:, :, :, 0]
        elif self.idx_mode == "sparse_row_std":
            sparse_attention_map_std = sparse_attention_map.masked_select(mask != 0).view(
                sparse_attention_map.shape[:-1] + (self.K,))
            attention_point_score = torch.std(sparse_attention_map_std, dim=-1)
        elif self.idx_mode == "sparse_col_sum":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2)
        elif self.idx_mode == "sparse_col_avg":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2) / sparse_num
        elif self.idx_mode == "sparse_col_sqr":
            attention_point_score = torch.sum(sparse_attention_map, dim=-2) / sparse_num / sparse_num
        else:
            raise ValueError('Please check the setting of idx mode!')
        idx = attention_point_score.topk(self.M, dim=-1)[1]
        return idx, attention_point_score, sparse_attention_map, mask

    def bin_conv(self, x):
        bin_prob_edge = self.bin_conv1(x)  # bin_prob_edge.shape == (B, num_bins/2, N)
        x = torch.cat((x, bin_prob_edge), dim=1)  # x.shape == (B, C+num_bins/2, N)
        x = self.bin_conv2(x)  # x.shape == (B, C, N)

        # TODO try Max
        bin_prob_edge = torch.max(bin_prob_edge, dim=-1, keepdim=True)[0]  # bin_prob_edge.shape == (B, num_bins/2, 1)
        bin_prob_edge = bin_prob_edge.permute(0, 2, 1)  # bin_prob_edge.shape == (B, 1, num_bins/2)
        bin_prob_edge = bin_prob_edge / self.scaling_factor
        bin_prob_edge = ops.norm_range(bin_prob_edge, dim=-1, n_min=0.5, n_max=1, mode=self.bin_norm_mode)
        bin_prob_inner = torch.flip((1 - bin_prob_edge), dims=(-1,))
        bin_prob = torch.cat((bin_prob_edge, bin_prob_inner), dim=-1)  # bin_prob.shape == (B, 1, num_bins)
        bin_prob = bin_prob.squeeze(1)  # bin_prob.shape == (B, num_bins)
        return x, bin_prob

    def bin_idx_selection(self):
        aps_chunks, idx_chunks = ops.sort_chunk(self.attention_point_score, self.num_bins, dim=-1, descending=True)
        # aps_chunks.shape == num_bins * (B, H, N/num_bins), # idx_sorted.shape == num_bins * (B, H, N/num_bins)
        B, H, chunk_size = aps_chunks[0].shape
        assert H == 1, "Number of heads should be 1!"

        idx_batch_list = []
        k_batch_list = []
        for i in range(B):
            k_list = []
            idx_list = []
            for j in range(self.num_bins):
                # each bin has K samples
                if j != self.num_bins - 1:
                    k = int(2 * self.M / self.num_bins * self.bin_prob[i, j])
                else:
                    k = self.M - sum(k_list)
                k_list.append(k)

                if self.bin_sample_mode == "topk":
                    idx_tmp = aps_chunks[j][i].topk(k, dim=-1)[1]  # idx.shape == (H, k)
                elif self.bin_sample_mode == "uniform":
                    idx_tmp = torch.randperm(chunk_size)[:k]
                    idx_tmp = idx_tmp.unsqueeze(0).expand(H, -1).to(self.attention_point_score.device)
                elif self.bin_sample_mode == "random":
                    if k == 0:
                        continue
                    aps_chunks_tmp = ops.norm_range(aps_chunks[j][i], dim=-1, n_min=0, n_max=1, mode="minmax")
                    aps_chunks_tmp = aps_chunks_tmp / (self.boltzmann_T + 1e-8)
                    aps_chunks_tmp = F.softmax(aps_chunks_tmp, dim=-1)
                    idx_tmp = torch.multinomial(aps_chunks_tmp, num_samples=k, replacement=False)
                else:
                    raise ValueError(
                        'Please check the setting of bin sample mode. It must be topk, multinomial or random!')

                idx = torch.gather(idx_chunks[j][i], dim=-1, index=idx_tmp)  # idx.shape == (H, k)
                idx_list.append(idx)
            idx_single = torch.cat(idx_list, dim=-1)  # idx_list.shape == (H, M)
            idx_batch_list.append(idx_single)
            k_single = torch.tensor(k_list).unsqueeze(0).expand(H, -1).to(self.attention_point_score.device)
            k_batch_list.append(k_single)
        idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)
        k_batch = torch.stack(k_batch_list, dim=0)  # k_batch.shape == (B, num_bins)
        return idx_batch, k_batch

    def bin2_idx_selection(self):
        # self.attention_point_score.shape == (B, H, N)
        aps_bins, idx_bins = ops.sort_chunk(self.attention_point_score, self.num_bins, dim=-1, descending=True)
        # aps_bins.shape == num_bins * (B, H, N/num_bins), # idx_sorted.shape == num_bins * (B, H, N/num_bins)
        B, H, bin_size = aps_bins[0].shape
        _, _, bin_size_min = aps_bins[-1].shape
        assert H == 1, "Number of heads should be 1!"
        aps_bins = torch.stack(aps_bins)  # aps_bins.shape == ( num_bins, B, H, N/num_bins)
        aps_bins = aps_bins.permute(1, 2, 0, 3)  # aps_bins.shape == (B, H, num_bins, N/num_bins)
        aps_bins = torch.mean(aps_bins, dim=-1)  # aps_bins.shape == (B, H, num_bins)
        aps_bins = ops.norm_range(aps_bins, dim=-1, n_min=0, n_max=1, mode="minmax")
        aps_bins /= (self.boltzmann_T + 1e-8)
        aps_bins = F.softmax(aps_bins, dim=-1)  # aps_bins.shape == (B, H, num_bins)

        idx_batch_list = []
        k_batch_list = []
        for i in range(B):
            idx_bin = torch.multinomial(aps_bins[i, 0], num_samples=self.M, replacement=True)
            # adjust the number of samples in each bin
            count_list = []
            rest_count = 0
            for k in range(self.num_bins):
                count = torch.eq(idx_bin, k)
                count = torch.sum(count) + rest_count
                if k == self.num_bins - 1:
                    bin_size_tmp = bin_size_min
                else:
                    bin_size_tmp = bin_size
                if count > bin_size_tmp:
                    rest_count = count - bin_size_tmp
                    count = bin_size_tmp
                else:
                    rest_count = 0
                count_list.append(count)
            if rest_count > 0:
                for k, count in enumerate(count_list):
                    count += rest_count
                    if count > bin_size:
                        rest_count = count - bin_size
                        count_list[k] = bin_size
                    else:
                        rest_count = 0
                        break
            idx_list = []
            for k in range(self.num_bins):
                if k == self.num_bins - 1:
                    bin_size_tmp = bin_size_min
                else:
                    bin_size_tmp = bin_size
                idx_tmp = torch.randperm(bin_size_tmp)[:count_list[k]]
                idx_tmp = idx_tmp.unsqueeze(0).to(self.attention_point_score.device)
                idx = torch.gather(idx_bins[k][i], dim=-1, index=idx_tmp)  # idx.shape == (H, k)
                idx_list.append(idx)
            idx_single = torch.cat(idx_list, dim=-1)  # idx_list.shape == (H, M)
            idx_batch_list.append(idx_single)
            k_single = torch.tensor(count_list).to(self.attention_point_score.device)
            k_batch_list.append(k_single)
        idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)
        k_batch = torch.stack(k_batch_list, dim=0)  # k_batch.shape == (B, num_bins)
        self.bin_prob = k_batch / self.M
        return idx_batch, k_batch

    def boltzmann_idx_selection(self):
        B, H, N = self.attention_point_score.shape
        aps_boltz = ops.norm_range(self.attention_point_score, dim=-1, n_min=0, n_max=1, mode=self.boltzmann_norm_mode)
        aps_boltz /= self.boltzmann_T
        aps_boltz = F.softmax(aps_boltz, dim=-1)
        idx_batch_list = []
        for i in range(B):
            idx_boltz_list = []
            for j in range(H):
                idx_boltz = torch.multinomial(aps_boltz[i, j], num_samples=self.M, replacement=False)
                idx_boltz_list.append(idx_boltz)
            idx_single = torch.stack(idx_boltz_list, dim=0)
            idx_batch_list.append(idx_single)
        idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)
        return idx_batch
