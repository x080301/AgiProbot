from torch import nn
import torch
import einops

from utils import ops
import time


def get_sparse_attention_map(x, K, attention_map):
    mask = ops.neighbor_mask(x, K)
    mask = mask.unsqueeze(1).expand(-1, attention_map.shape[1], -1, -1)
    # print(f'attention_map.shape{attention_map.shape}')
    # print(f'mask.shape{mask.shape}')
    sparse_attention_map = attention_map * mask
    return mask, sparse_attention_map


def bin_idx_selection(attention_point_score, num_bins, bin_prob, M, bin_sample_mode):
    bin_prob = bin_prob.clone().detach()
    # self.attention_point_score.shape == (B, H, N)
    aps_chunks, idx_chunks = ops.sort_chunk(attention_point_score, num_bins, dim=-1, descending=True)
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

            if bin_sample_mode == "topk":
                idx_tmp = aps_chunks[j][i].topk(k, dim=-1)[1]  # idx.shape == (H, k)
            elif bin_sample_mode == "uniform":
                idx_tmp = torch.randperm(chunk_size)[:k]
                idx_tmp = idx_tmp.unsqueeze(0).expand(H, -1).to(attention_point_score.device)
            elif bin_sample_mode == "random":
                if k == 0:
                    continue
                aps_chunks_tmp = ops.norm_range(aps_chunks[j][i], dim=-1, n_min=0, n_max=1, mode="minmax")
                aps_chunks_tmp = aps_chunks_tmp
                aps_chunks_tmp = torch.nn.functional.softmax(aps_chunks_tmp, dim=-1)
                idx_tmp = torch.multinomial(aps_chunks_tmp, num_samples=k, replacement=False)
            else:
                raise ValueError(
                    'Please check the setting of bin sample mode. It must be topk, multinomial or random!')
            idx = torch.gather(idx_chunks[j][i], dim=-1, index=idx_tmp)  # idx.shape == (H, k)
            idx_list.append(idx)
        idx_single = torch.cat(idx_list, dim=-1)  # idx_list.shape == (H, M)
        idx_batch_list.append(idx_single)
        k_single = torch.tensor(k_list).to(attention_point_score.device)
        k_batch_list.append(k_single)
    idx_batch = torch.stack(idx_batch_list, dim=0)  # idx_batch.shape == (B, H, M)
    k_batch = torch.stack(k_batch_list, dim=0)  # k_batch.shape == (B, num_bins)
    return idx_batch, k_batch, idx_chunks


def calculate_num_points_to_choose(bin_prob, max_num_points, total_points_to_choose):
    """

    :param total_points_to_choose: Int
    :param bin_prob: torch.Tensor(B,num_bins)
    :param max_num_points: torch.Tensor(B,num_bins)
    :return: number of choosen points, torch.Tensor(B,num_bins)
    """

    B, num_bins = bin_prob.shape
    bin_prob += 1e-8

    num_chosen_points_in_bin = torch.zeros_like(bin_prob, device=bin_prob.device)
    for _ in range(num_bins):
        bin_prob = bin_prob / torch.sum(bin_prob, dim=1, keepdim=True)
        num_to_choose = total_points_to_choose - torch.sum(num_chosen_points_in_bin, dim=1, keepdim=True)
        # print(torch.max(num_to_choose))

        num_chosen_points_in_bin += bin_prob * num_to_choose
        num_chosen_points_in_bin = torch.where(num_chosen_points_in_bin >= max_num_points, max_num_points,
                                               num_chosen_points_in_bin)
        bin_prob = bin_prob * torch.where(num_chosen_points_in_bin >= max_num_points, 0, 1)

    num_chosen_points_in_bin = num_chosen_points_in_bin.int()
    # print(torch.argmax(max_num_points - num_chosen_points_in_bin, dim=1).shape)

    num_chosen_points_in_bin[
        torch.arange(0, B), torch.argmax(max_num_points - num_chosen_points_in_bin,
                                         dim=1)] += total_points_to_choose - torch.sum(num_chosen_points_in_bin, dim=1)

    # print(num_chosen_points_in_bin)
    # print(torch.sum(num_chosen_points_in_bin, dim=1))
    # print(max_num_points)
    return num_chosen_points_in_bin


def calculate_num_points_to_choose_one_iteration(probability, max_num_points, num_undecided_points,
                                                 num_points_in_bins):  # , total_points):
    """

    :param num_undecided_points: torch.Tensor(B,)
    :param probability: torch.Tensor(B,num_bins)
    :param max_num_points: torch.Tensor(B,num_bins)
    :return: number of choosen points, torch.Tensor(B,num_bins);
    """
    num_undecided_points = num_undecided_points.reshape(-1, 1)

    num_points_to_choose = probability * num_points_in_bins
    num_points_to_choose = num_points_to_choose * num_undecided_points / torch.sum(num_points_to_choose,
                                                                                   dim=1, keepdim=True)
    num_points_to_choose = num_points_to_choose.int()

    num_points_to_choose = torch.where(num_points_to_choose < max_num_points, num_points_to_choose, max_num_points)
    # print(f'num_points_to_choose:{num_points_to_choose}')

    return num_points_to_choose


def nonuniform_bin_idx_selection(attention_point_score, bin_boundaries, bin_prob, normalization_mode, M,
                                 bin_sample_mode):
    bin_prob = bin_prob.clone().detach()
    # bin_prob.shape == (B, num_bins)
    # self.attention_point_score.shape == (B, H, N)
    aps_chunks, idx_chunks = ops.sort_chunk_nonuniform(attention_point_score, bin_boundaries, normalization_mode)
    # print(f'idx.dtype3:{idx_chunks[0][0].dtype}')
    # aps_chunks.shape == num_bins * (B, H, n), # idx_sorted.shape == num_bins * (B, H, N/num_bins)
    num_bins = bin_boundaries[0].nelement()
    B, H, N = attention_point_score.shape

    # chunk_size = aps_chunks[j][i].shape[1]
    assert H == 1, "Number of heads should be 1!"

    max_num_points = torch.zeros((B, num_bins), dtype=torch.long, device=bin_prob.device)
    for i in range(B):
        for j in range(num_bins):
            max_num_points[i, j] = aps_chunks[j][i].shape[1]
    # print(f' bin_prob{bin_prob}-----------')

    k_point_to_choose = calculate_num_points_to_choose(bin_prob, max_num_points, M)

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
                idx_tmp = idx_tmp.unsqueeze(0).expand(H, -1).to(attention_point_score.device)
            elif bin_sample_mode == "random":
                if k != 0:
                    aps_chunks_tmp = ops.norm_range(aps_chunks[j][i], dim=-1, n_min=0, n_max=1, mode="minmax")
                    aps_chunks_tmp = torch.nn.functional.softmax(aps_chunks_tmp, dim=-1)
                    # print(f'k:{k}')
                    # print(f'aps_chunks_tmp.shape:{aps_chunks_tmp.shape}')
                    if aps_chunks_tmp.nelement() < k:
                        print(f'aps_chunks_tmp{aps_chunks_tmp.nelement()},k{k}')
                        exit(-1)
                    idx_tmp = torch.multinomial(aps_chunks_tmp, num_samples=k, replacement=False)
            else:
                raise ValueError(
                    'Please check the setting of bin sample mode. It must be topk, multinomial or random!')
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
    # print(f'idx.dtype2:{idx.dtype}')

    return idx_batch, k_point_to_choose, idx_chunks


class DownSampleToken(nn.Module):
    def __init__(self, config_ds, layer):
        super(DownSampleToken, self).__init__()

        self.K = 32
        self.num_bins = config_ds.bin.num_bins[layer]
        self.bin_mode = config_ds.bin.mode[layer]
        self.bin_sample_mode = config_ds.bin.sample_mode[layer]
        self.num_heads = config_ds.num_heads[layer]
        self.normalization_mode = config_ds.bin.normalization_mode[layer]
        self.M = config_ds.M[layer]
        self.res = config_ds.res.enable[layer]

        bin_boundaries_upper = [float('inf')]
        bin_boundaries_upper.extend(config_ds.bin.bin_boundaries[layer])
        bin_boundaries_lower = config_ds.bin.bin_boundaries[layer]
        bin_boundaries_lower.append(float('-inf'))
        self.bin_boundaries = [torch.asarray(bin_boundaries_upper).reshape(1, 1, 1, self.num_bins),
                               # [inf, 0.503, 0.031, -0.230, -0.427, -0.627]
                               torch.asarray(bin_boundaries_lower).reshape(1, 1, 1, self.num_bins)
                               # [0.503, 0.031, -0.230, -0.427, -0.627, -inf]
                               ]

        q_in = config_ds.q_in[layer]
        q_out = config_ds.q_out[layer]
        k_in = config_ds.k_in[layer]
        k_out = config_ds.k_out[layer]
        v_in = config_ds.v_in[layer]
        v_out = config_ds.v_out[layer]

        self.bin_tokens = nn.Parameter(torch.randn(1, q_in, self.num_bins))

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.bin_prob = None
        self.idx_chunks = None
        self.idx = None
        self.attention_point_score = None

    def forward(self, x, x_xyz=None):

        # self.bin_tokens = self.bin_tokens.to(x.device)
        # x.shape == (B, C, N)
        B, C, N = x.shape
        # print(f'x.shape{self.bin_tokens.device}')
        bin_tokens = einops.repeat(self.bin_tokens, '1 c num_bins -> b c num_bins', b=B)
        # bin_tokens.shape ==(B,C,num_bins)

        # print(f'x.shape{x.device}')
        # print(f'bin_tokens.shape{bin_tokens.device}')
        x_and_token = torch.concat((x, bin_tokens), dim=2)  # x_and_token: (B,C,N+num_bins)
        # print(f'x_and_token.shape{x_and_token.device}')

        if self.num_heads == 1:
            q = self.q_conv(x).unsqueeze(dim=1)  # q.shape == (B, 1, C, N)
            k = self.k_conv(x_and_token).unsqueeze(dim=1)  # k.shape ==  (B, 1, C, N+num_bins)
            v = self.v_conv(x_and_token).unsqueeze(dim=1)  # v.shape ==  (B, 1, C, N+num_bins)

            q = einops.rearrange(q, 'b 1 c n -> b 1 n c')  # q: (B, 1, N, C)

            energy = q @ k  # energy: (B,1,N,N+num_bins)

            attention_map = self.softmax(energy)  # attention_map: (B,1,N,N+num_bins)

            attention_points, attention_bins = torch.split(attention_map, [N, self.num_bins], dim=-1)
            # energy_points: (B,1,H,N,N)
            # energy_bins: (B,1,H,N,num_bins)

            bin_prob, _ = torch.max(attention_bins, dim=-2)  # x_bins: (B,1,num_bins)
            bin_prob = bin_prob.squeeze(1)  # x_bins: (B,num_bins)
            # print(f'bin_prob.shape:{bin_prob.shape}')
        else:
            raise NotImplementedError

        mask, sparse_attention_map = get_sparse_attention_map(x, self.K, attention_points)
        sparse_num = torch.sum(mask, dim=-2) + 1e-8
        attention_point_score = torch.sum(sparse_attention_map, dim=-2) / sparse_num / sparse_num

        if self.bin_mode == 'uniform_split_bin':
            idx_down, _, idx_chunks = bin_idx_selection(attention_point_score, self.num_bins,
                                                        bin_prob, self.M, self.bin_sample_mode)
        elif self.bin_mode == 'nonuniform_split_bin':

            idx_down, k_point_to_choose, idx_chunks = nonuniform_bin_idx_selection(attention_point_score,
                                                                   self.bin_boundaries,
                                                                   bin_prob,
                                                                   self.normalization_mode,
                                                                   self.M,
                                                                   self.bin_sample_mode)
        else:
            raise NotImplementedError

        attention_down = torch.gather(attention_map, dim=2,
                                      index=idx_down.unsqueeze(3).expand(-1, -1, -1, attention_map.shape[-1]))
        # attention_map: (B,H,N,N+num_bins)
        # attention_down: (B,H,M,N+num_bins)
        v_down = (attention_down @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # attention_down: (B,H,M,N+num_bins)
        # v.permute(0, 1, 3, 2): (B, H, N+num_bins, C)
        # v_down.shape == (B, M, H, C)
        x_ds = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)
        # v_down.shape == (B, C, M)
        if self.res:
            x_ds = self.res_block(x, x_ds, idx_down)

        self.bin_prob = bin_prob
        self.idx_chunks = idx_chunks
        self.idx = idx_down
        self.attention_point_score = attention_point_score
        self.k_point_to_choose = k_point_to_choose

        return (x_ds, idx_down), (None, None)

    def output_variables(self, *args):
        # 'attention_point_score'
        # 'idx'
        # 'idx_chunks'
        # 'bin_prob'

        # print(vars().keys())
        variables = None
        for i, key in enumerate(args):
            if i == 0:
                variables = getattr(vars()['self'], key)
                # variables = vars()[f'self.{key}']
            elif i == 1:
                variables = (variables,) + (getattr(vars()['self'], key),)
                # variables = (variables,) + (vars()[f'self.{key}'],)
            else:
                variables = variables + (getattr(vars()['self'], key),)
                # variables = variables + (vars()[f'self.{key}'],)

        return variables

    def res_block(self, x, x_ds, idx):  # x.shape == (B, C, N), x_ds.shape == (B, C, M)
        x_tmp = torch.gather(x, dim=-1, index=idx)  # x_res.shape == (B, 1, M)
        x_res = self.bn1(x_ds + x_tmp)  # x_res.shape == (B, C, M)
        if self.ff == True:
            x_tmp = self.ffn(x_res)
            x_res = self.bn2(x_ds + x_tmp)
        return x_res  # x_res.shape == (B, C, M)
