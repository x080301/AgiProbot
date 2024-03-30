import torch


def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, M, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a ** 2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b ** 2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    # TODO: some values inside pairwise_distance is positive
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    distance, idx = pairwise_distance.topk(k=k, dim=-1)  # idx.shape == (B, N, K)
    return distance, idx


def select_neighbors(pcd, K, neighbor_type, normal_channel=False):  # pcd.shape == (B, C, N)
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    if normal_channel and pcd.shape[-1] == 6:
        _, idx = knn(pcd[:, :, :3], pcd[:, :, :3], K)  # idx.shape == (B, N, K, 3)
    else:
        _, idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
    if neighbor_type == 'neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors, idx  # neighbors.shape == (B, C, N, K), idx.shape == (B, N, K)


def select_neighbors_interpolate(unknown, known, known_feature, K=3):
    known = known.permute(0, 2, 1)  # known.shape == (B, M, C)
    known_feature = known_feature.permute(0, 2, 1)  # known_feature.shape == (B, M, C)
    unknown = unknown.permute(0, 2, 1)  # unknown.shape == (B, N, C)
    d, idx = knn(unknown, known, K)  # idx.shape == (B, N, K)
    d = -1 * d  # d.shape == (B, N, K)
    neighbors = index_points(known_feature, idx)  # neighbors.shape == (B, N, K, C)
    neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    return neighbors, idx, d  # neighbors.shape == (B, C, N, K), idx.shape == (B, N, K), d.shape == (B, N, K)


def group(pcd, K, group_type, normal_channel=False):
    if group_type == 'neighbor':
        neighbors, idx = select_neighbors(pcd, K, 'neighbor', normal_channel)  # neighbors.shape == (B, C, N, K)
        output = neighbors  # output.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff, idx = select_neighbors(pcd, K, 'diff', normal_channel)  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors, idx = select_neighbors(pcd, K, 'neighbor', normal_channel)  # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff, idx = select_neighbors(pcd, K, 'diff', normal_channel)  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output, idx


def l2_global(q, k):  # q.shape == (B, H, N, D), k.shape == (B, H, D, N)
    inner = -2 * torch.matmul(q, k)  # inner.shape == (B, H, N, N)
    qq = torch.sum(q ** 2, dim=-1, keepdim=True)  # qq.shape == (B, H, N, 1)
    kk = torch.sum(k.transpose(-2, -1) ** 2, dim=-1, keepdim=True)  # kk.shape == (B, H, N, 1)
    qk_l2 = qq + inner + kk.transpose(-2, -1)  # qk_l2.shape == (B, H, N, N)
    return qk_l2


def neighbor_mask(pcd, K):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    _, idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    B, N, _ = idx.shape
    mask = torch.zeros(B, N, N, dtype=torch.float32, device=idx.device)  # mask.shape == (B, N, N)
    mask.scatter_(2, idx, 1.0)
    return mask


def gather_by_idx(pcd, idx):
    # pcd.shape == (B, C, N)
    # idx.shape == (B, H, K)
    B, C, N = pcd.shape
    _, _, K = idx.shape
    idx = idx.expand(-1, C, -1)
    # output = torch.zeros(B, C, K, dtype=torch.float32, device=pcd.device)
    # output.scatter_(2, idx, pcd) # output.shape == (B, C, K)
    output = torch.gather(pcd, 2, idx)  # output.shape == (B, C, K)
    return output


def norm_range(x, dim=-1, n_min=0, n_max=1, mode="minmax"):
    if mode == 'minmax':
        x_norm = (x - torch.min(x, dim=dim, keepdim=True)[0]) / (
                torch.max(x, dim=dim, keepdim=True)[0] - torch.min(x, dim=dim, keepdim=True)[0] + 1e-8)
    elif mode == 'sigmoid':
        x_norm = torch.sigmoid(x)
    elif mode == 'tanh':
        x_norm = torch.tanh(x)
        x_norm = (x_norm + 1.) / 2
    elif mode == "z-score":
        miu = n_min
        x_norm = (x - torch.mean(x, dim=dim, keepdim=True)) / torch.std(x, dim=dim, unbiased=False, keepdim=True) + miu
        return x_norm
    else:
        raise ValueError(f'norm_range mode should be minmax, sigmoid or tanh, but got {mode}')
    x_norm = x_norm * (n_max - n_min) + n_min
    return x_norm


def update_sampling_score_bin_boundary(old_bin_boundaries, attention_point_score, num_bins, momentum_update_factor):
    # old_bin_boundaries:2 * (1,1,1,num_bins)
    # attention_point_score: (B, H, N)

    num_sampling_scores = attention_point_score.nelement()

    bin_boundaries_index = torch.arange(1, num_bins) / num_bins * num_sampling_scores
    bin_boundaries_index = bin_boundaries_index.to(attention_point_score.device).int()

    sorted_scores, _ = torch.sort(attention_point_score.flatten(), dim=0, descending=True)
    bin_boundaries = sorted_scores[bin_boundaries_index]

    torch.distributed.all_reduce(bin_boundaries)  # , reduce_op=torch.distributed.ReduceOp.SUM)
    bin_boundaries = bin_boundaries / torch.distributed.get_world_size()

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

        new_bin_boundaries = [torch.asarray(bin_boundaries_upper).reshape(1, 1, 1, num_bins),
                              # [inf, 0.503, 0.031, -0.230, -0.427, -0.627]
                              torch.asarray(bin_boundaries_lower).reshape(1, 1, 1, num_bins)
                              # [0.503, 0.031, -0.230, -0.427, -0.627, -inf]
                              ]

        # print(f'new_bin_boundaries:{new_bin_boundaries}')
    return new_bin_boundaries


def sort_chunk_nonuniform(attention_point_score, bin_boundaries, num_bins, normalization_mode,
                          dynamic_boundaries_enable, momentum_update_factor):
    """

    :param attention_point_score: (B,1,N)
    :param bin_boundaries: list with size num_bins-1
    :return: x_chunks, idx_chunks, list[list[torch.Tensor(n,)]],with descending order, num_bins*B*(n,)
    """

    # num_bins = bin_boundaries[0].nelement()
    B, H, N = attention_point_score.shape
    # print(f'B{B},H{H},N{N}')
    # bin_boundaries = [item.to(attention_point_score.device) for item in bin_boundaries]

    if normalization_mode == 'no_normalization':
        pass
    elif normalization_mode == 'z_score':
        # attention_point_score: (B,1,N)
        attention_point_score = (attention_point_score - torch.mean(attention_point_score, dim=2, keepdim=True)) \
                                / torch.std(attention_point_score, dim=2, unbiased=False, keepdim=True)

    elif normalization_mode == 'z_score_no_std':
        attention_point_score = torch.log(attention_point_score)
        # try:
        #     attention_point_score = torch.log(attention_point_score)
        # except:
        #     print(f'----------Error in log-----------------')
        #     print(f'attention_point_score:\n{attention_point_score}')
        #     print(f'zero or negative value exists = {torch.min(attention_point_score).item() <= 0}')
        #     print(f'minimun is {torch.min(attention_point_score).item()}')

        # attention_point_score = attention_point_score - torch.mean(attention_point_score, dim=2, keepdim=True)
        attention_point_score_no_infnan = torch.where((attention_point_score == float('-inf')) |
                                                      (attention_point_score == float('inf')) |
                                                      torch.isnan(attention_point_score), 0, attention_point_score)
        attention_point_score = attention_point_score - torch.mean(attention_point_score_no_infnan, dim=2, keepdim=True)
        attention_point_score = torch.where((attention_point_score == float('inf')), 100, attention_point_score)
        attention_point_score = torch.where(torch.isnan(attention_point_score), 0, attention_point_score)

    attention_point_score = attention_point_score.reshape(B, H, N, 1)
    # bin_boundaries: [(1,1,1,6),(1,1,1,6)]

    if dynamic_boundaries_enable:
        bin_boundaries = update_sampling_score_bin_boundary(bin_boundaries, attention_point_score, num_bins,
                                                            momentum_update_factor)

    bin_points_mask = (attention_point_score < bin_boundaries[0]) & (attention_point_score >= bin_boundaries[1])
    # bin_points_mask: (B,H,N,num_bins)
    index_batch, _, index_point, index_bin = torch.where(bin_points_mask)

    idx_chunks = [[index_point[(index_bin == i) & (index_batch == j)].reshape(1, -1)
                   for j in range(B)]
                  for i in range(num_bins)]
    x_chunks = [[attention_point_score[j, 0, :][index_point[(index_bin == i) & (index_batch == j)]].reshape(1, -1)
                 for j in range(B)]
                for i in range(num_bins)]

    # num_points_in_bins = torch.zeros(B, num_bins)
    # for i in range(num_bins):
    #     for j in range(B):
    #         num_points_in_bins[j, i] = idx_chunks[i][j].nelement()

    # idx_chunks: num_bins * B *(H, n)
    # x_chunks: num_bins * B *(H, n)

    # for i in range(num_bins):
    #     x_chunks_one_bin = []
    #     idx_chunks_one_bin = []
    #     for j in range(B):
    #         if i == 0:
    #             index_in_bin = torch.where(attention_point_score[j, 0, :] > bin_boundaries[i])[0]
    #         elif i < num_bins - 1:
    #             index_in_bin = torch.where(
    #                 (attention_point_score[j, 0, :] > bin_boundaries[i]) & (
    #                         attention_point_score[j, 0, :] < bin_boundaries[i - 1]))[0]
    #         else:
    #             index_in_bin = torch.where(attention_point_score[j, 0, :] < bin_boundaries[i - 1])[0]
    #
    #         x_chunks_one_bin.append(attention_point_score[j, 0, :][index_in_bin].reshape(1, -1))
    #         idx_chunks_one_bin.append(index_in_bin.reshape(1, -1))
    #
    #         # print(f'idex_in_bin{j} == {len(index_in_bin)}')

    # print(f'idx.dtype4:{index_in_bin.dtype}')
    # exit(-1)
    return x_chunks, idx_chunks, bin_boundaries, bin_points_mask

    # z_normalized_x = (attention_point_score - torch.mean(attention_point_score, dim=2, keepdim=True))
    # # z_normalized_x.shape = (B,1,N)
    # topk_values, _ = torch.topk(z_normalized_x, k=int(z_normalized_x.shape[2] * 0.0228), dim=2, largest=True)
    # max_value_9772 = topk_values[:, :, -1]
    # # max_value_9772.shape = (B,1)
    # topk_values, _ = torch.topk(-z_normalized_x, k=int(z_normalized_x.shape[2] * 0.0228), dim=2, largest=True)
    # min_value_0228 = topk_values[:, :, -1]
    # # print(f'min_value_0228.shape={min_value_0228.shape}')
    # # min_value_0228.shape = (B,1)
    # bin_width = (max_value_9772 - min_value_0228) / num_bins
    #
    # x_chunks = []
    # idx_chunks = []
    # for i in range(num_bins):
    #     x_chunks_bin_i = []
    #     idx_chunks_bin_i = []
    #     for b in range(attention_point_score.shape[0]):
    #         if i == 0:
    #             indices = z_normalized_x[b, 0, :] > (max_value_9772 - bin_width)[b]
    #         elif i != num_bins - 1:
    #             indices = (z_normalized_x[b, 0, :] <= (max_value_9772 - i * bin_width)[b]) & (
    #                     z_normalized_x[b, 0, :] > (max_value_9772 - i * bin_width - bin_width)[b])
    #         else:  # i=num_bins - 1
    #             indices = z_normalized_x[b, 0, :] <= (max_value_9772 - i * bin_width)[b]
    #
    #         idx_chunks_bin_i_b = torch.nonzero(indices)
    #         # print(f'idx_chunks_bin_i_b.shape={idx_chunks_bin_i_b}')
    #         num_points_in_bin_i = idx_chunks_bin_i_b.shape[0]
    #         # print(f'num_points_in_bin: {num_points_in_bin_i}')
    #
    #         idx_chunks_bin_i.append(idx_chunks_bin_i_b.reshape(1, num_points_in_bin_i))
    #         x_chunks_bin_i.append(attention_point_score[b, 0, idx_chunks_bin_i_b].reshape(1, num_points_in_bin_i))
    #
    #     x_chunks.append(x_chunks_bin_i)
    #     idx_chunks.append(idx_chunks_bin_i)


def sort_chunk(attention_point_score, num_bins, dim=-1, descending=False):
    """

    :param attention_point_score: torch.Tensor (B,1,N)
    :param num_bins: int
    :param dim: int
    :param descending: bool
    :param bin_split_mode: str, 'uniform' or 'nonuniform'
    :return: tuple or list of torch.Tensors (B,1,n).
    """
    x_sorted, idx_sorted = torch.sort(attention_point_score, dim=dim, descending=descending)
    x_chunks = torch.chunk(x_sorted, num_bins, dim=dim)  # x_chunks.shape == num_bins * (B, H, N/num_bins)
    idx_chunks = torch.chunk(idx_sorted, num_bins, dim=dim)  # idx_sorted.shape == num_bins * (B, H, N/num_bins)

    return x_chunks, idx_chunks


def reshape_gathered_variable(gathered_variable):
    # if isinstance(gathered_variable[0], torch.Tensor):
    #     if len(gathered_variable[0].shape)==4:
    #         # gathered_variable: num_layers * (B, num_bins, H, n)
    #         num_layers = len(gathered_variable)
    #         B = len(gathered_variable[0])
    #
    #         gathered_variable_in_batches = []
    #         for i in range(B):
    #             gathered_variable_in_one_batch = []
    #             for j in range(num_layers):
    #                 gathered_variable_in_one_batch.append(gathered_variable[j][i])
    #             # gathered_variable_in_one_batch: num_layers * (num_bins, H, n)
    #             gathered_variable_in_batches.append(gathered_variable_in_one_batch)
    #         # gathered_variable_in_batches: B * num_layers * (num_bins, H, n)
    #         gathered_variable = gathered_variable_in_batches
    #     else:
    #         # gathered_variable: num_layers * (B, H, N) or num_layers * (B, num_bins)
    #         gathered_variable = torch.stack(gathered_variable, dim=0)
    #         gathered_variable = gathered_variable.transpose(0, 1).contiguous()
    #         # gathered_variable: (B, num_layers, H, N) or (B, num_layers, num_bins)
    # else:

    # gathered_variable:
    # num_layers * B * num_bins * (H,n) or
    # num_layers * (B, num_bins, H, n) or
    # num_layers * (B, H, N) or
    # num_layers * (B, num_bins)
    num_layers = len(gathered_variable)
    B = len(gathered_variable[0])

    gathered_variable_in_batches = []
    for i in range(B):
        gathered_variable_in_one_batch = []
        for j in range(num_layers):
            gathered_variable_in_one_batch.append(gathered_variable[j][i])
        # gathered_variable_in_one_batch: num_layers * num_bins * (H,n)
        gathered_variable_in_batches.append(gathered_variable_in_one_batch)
    # gathered_variable_in_batches: B * num_layers * num_bins * (H,n)
    gathered_variable = gathered_variable_in_batches

    return gathered_variable
    # return:
    # B * num_layers * num_bins * (H,n) or
    # B * num_layers * (num_bins, H, n) or
    # B * num_layers * (H, N) or
    # B * num_layers * (num_bins)


def gather_variable_from_gpus(downsample_module, variable_name, rank, world_size, device):
    variable_to_gather = downsample_module.output_variables(variable_name)

    if isinstance(variable_to_gather, torch.Tensor):
        variable_gather_list = [torch.empty_like(variable_to_gather).to(device) for _ in
                                range(world_size)]
        torch.distributed.all_gather(variable_gather_list, variable_to_gather)

        if rank == 0:
            return torch.concat(variable_gather_list, dim=0)

    else:
        # variable_to_gather: num_bins * (B,H,n) or num_bins * B * (H,n)

        if isinstance(variable_to_gather[0], torch.Tensor):
            variable_to_gather = torch.stack(variable_to_gather, dim=0)
            variable_to_gather = variable_to_gather.permute(1, 0, 2, 3).contiguous()
            # variable_to_gather: (B,num_bins,H,n)
            variable_gather_list = [torch.empty_like(variable_to_gather).to(device) for _ in
                                    range(world_size)]
            torch.distributed.all_gather(variable_gather_list, variable_to_gather)

            if rank == 0:
                return torch.concat(variable_gather_list, dim=0)
                # return: (B,num_bins,H,n)

        else:
            # variable_to_gather: num_bins * B * (H,n)

            num_bins = len(variable_to_gather)
            B = len(variable_to_gather[0])

            data_size = torch.empty((B, num_bins), device=variable_to_gather[0][0].device)
            variable_in_batches = []
            for i in range(B):
                for j in range(num_bins):
                    variable_in_batches.append(variable_to_gather[j][i].flatten())
                    data_size[i, j] = variable_to_gather[j][i].nelement()
            # variable_in_batches: (B * num_bins) * (n,)
            variable_in_batches = torch.concat(variable_in_batches, dim=0)
            # variable_in_batches: (B * num_bins * n),
            # print(f'variable_in_batches{variable_in_batches.shape}')
            # print(f'data_size{torch.sum(data_size)}')

            data_size_gather_list = [torch.empty_like(data_size).to(device) for _ in range(world_size)]
            torch.distributed.all_gather(data_size_gather_list, data_size)

            if rank == 0:
                variable_gather_list = [
                    torch.empty((int(torch.sum(data_size).item()),), dtype=variable_in_batches.dtype).to(device) for
                    data_size in data_size_gather_list]
            else:
                variable_gather_list = None

            # print(f'variable_in_batches:{variable_in_batches.dtype}')
            # print(f'variable_gather_list:{variable_gather_list[0].dtype}')
            torch.distributed.gather(variable_in_batches, gather_list=variable_gather_list, dst=0)
            # variable_gather_list: world_size * (B * num_bins * (n1+n2+n3+...))
            if rank == 0:
                variable_to_return = []
                for data_size, variable_in_batches in zip(data_size_gather_list, variable_gather_list):
                    # variable_in_batches: (B * num_bins * n),
                    # data_size: (B , num_bins)
                    begin_idex = 0
                    for i in range(B):
                        variable_in_one_batch = []
                        for j in range(num_bins):
                            end_index = begin_idex + int(data_size[i, j].item())
                            one_variable = variable_in_batches[begin_idex:end_index].reshape(1, -1)
                            begin_idex = end_index
                            variable_in_one_batch.append(one_variable)
                        # variable_in_one_batch: num_bins * (1,n)
                        variable_to_return.append(variable_in_one_batch)
                # variable_to_return: B * num_bins * (H,n)
                return variable_to_return


def calculate_num_points_to_choose(bin_prob, max_num_points, total_points_to_choose):
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
        num_chosen_points_in_bin = torch.where(num_chosen_points_in_bin >= max_num_points, max_num_points,
                                               num_chosen_points_in_bin)
        bin_prob = bin_prob * torch.where(num_chosen_points_in_bin >= max_num_points, 0, 1)

    num_chosen_points_in_bin = num_chosen_points_in_bin.int()
    # print(torch.argmax(max_num_points - num_chosen_points_in_bin, dim=1).shape)

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


def bin_partition(attention_point_score, bin_boundaries, dynamic_boundaries_enable, momentum_update_factor,
                  normalization_mode, num_bins):
    B, H, N = attention_point_score.shape

    if bin_boundaries is not None:
        bin_boundaries = [item.to(attention_point_score.device) for item in bin_boundaries]

    # print(f'B{B},H{H},N{N}')
    # bin_boundaries = [item.to(attention_point_score.device) for item in bin_boundaries]
    if normalization_mode == 'no_normalization':
        pass
    elif normalization_mode == 'z_score':
        # attention_point_score: (B,1,N)
        attention_point_score = (attention_point_score - torch.mean(attention_point_score, dim=2, keepdim=True)) \
                                / torch.std(attention_point_score, dim=2, unbiased=False, keepdim=True)

    else:
        raise NotImplementedError

    attention_point_score = attention_point_score.reshape(B, H, N, 1)
    # bin_boundaries: [(1,1,1,6),(1,1,1,6)]
    if dynamic_boundaries_enable:
        bin_boundaries = update_sampling_score_bin_boundary(bin_boundaries, attention_point_score, num_bins,
                                                            momentum_update_factor)
    bin_points_mask = (attention_point_score < bin_boundaries[0]) & (attention_point_score >= bin_boundaries[1])
    # bin_points_mask: (B,H,N,num_bins)
    return bin_boundaries, bin_points_mask


def generating_downsampled_index(M, attention_point_score, bin_points_mask, bin_sample_mode, boltzmann_T,
                                 k_point_to_choose):
    if bin_sample_mode == "topk":
        # attention_point_score: (B, H, N)
        attention_point_score = attention_point_score + 1e-8

        # bin_points_mask: (B, H, N, num_bins)
        masked_attention_point_score = attention_point_score.unsqueeze(3) * bin_points_mask
        # masked_attention_point_score: (B, H, N, num_bins)

        _, attention_index_score = torch.sort(masked_attention_point_score, dim=2, descending=True)
        attention_index_score = attention_index_score.squeeze(dim=1)
        # attention_index_score: (B, N, num_bins)

        B, _, N, num_bins = bin_points_mask.shape
        index_down = []
        for batch_index in range(B):
            sampled_index_in_one_batch = []
            for bin_index in range(num_bins):
                sampled_index_in_one_batch.append(
                    attention_index_score[batch_index, :k_point_to_choose[batch_index, bin_index], bin_index])
            index_down.append(torch.concat(sampled_index_in_one_batch))
        index_down = torch.stack(index_down).reshape(B, 1, M)
        # sampled_index: (B,H,M)

    elif bin_sample_mode == "uniform" or bin_sample_mode == "random":
        if bin_sample_mode == "uniform":
            sampling_probabilities = bin_points_mask.float().squeeze(dim=1)
        elif bin_sample_mode == "random":
            # attention_point_score: (B, H, N)
            # bin_points_mask: (B, H, N, num_bins)

            sampling_probabilities = torch.exp(attention_point_score.unsqueeze(3) / boltzmann_T) * bin_points_mask
            # sampling_probabilities = torch.exp(attention_point_score.unsqueeze(3) / 0.01) * bin_points_mask
            sampling_probabilities = sampling_probabilities / torch.sum(sampling_probabilities, dim=2, keepdim=True)
            sampling_probabilities = sampling_probabilities.squeeze(dim=1)
            # sampling_probabilities: (B,N,num_bins)

            sampling_probabilities[torch.isnan(sampling_probabilities)] = 1e-8

        B, N, num_bins = sampling_probabilities.shape

        sampling_probabilities = sampling_probabilities.permute(0, 2, 1).reshape(-1, N)
        # sampling_probabilities: (B*num_bins,N)

        sampled_index_M_points = torch.multinomial(sampling_probabilities, M)
        # sampled_index_M_points: (B*num_bins,M)
        sampled_index_M_points = sampled_index_M_points.reshape(B, num_bins, M)
        # sampled_index_M_points: (B,num_bins,M)

        index_down = []
        for batch_index in range(B):
            sampled_index_in_one_batch = []
            for bin_index in range(num_bins):
                sampled_index_in_one_batch.append(
                    sampled_index_M_points[batch_index, bin_index, :k_point_to_choose[batch_index, bin_index]])
            index_down.append(torch.concat(sampled_index_in_one_batch))
        index_down = torch.stack(index_down).reshape(B, 1, M)
        # sampled_index: (B,H,M)

    else:
        raise ValueError(
            'Please check the setting of bin sample mode. It must be topk, multinomial or random!')
    return index_down
