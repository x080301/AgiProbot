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
    aa = torch.sum(a**2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    # TODO: some values inside pairwise_distance is positive
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    distance, idx = pairwise_distance.topk(k=k, dim=-1)  # idx.shape == (B, N, K)
    return distance, idx


def select_neighbors(pcd, K, neighbor_type, normal_channel=False): # pcd.shape == (B, C, N)
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
    return neighbors, idx # neighbors.shape == (B, C, N, K), idx.shape == (B, N, K)
    
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
        neighbors, idx = select_neighbors(pcd, K, 'neighbor', normal_channel)   # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff, idx = select_neighbors(pcd, K, 'diff', normal_channel)  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output, idx

def l2_global(q, k): # q.shape == (B, H, N, D), k.shape == (B, H, D, N)
    inner = -2 * torch.matmul(q, k)  # inner.shape == (B, H, N, N)
    qq = torch.sum(q**2, dim=-1, keepdim=True)  # qq.shape == (B, H, N, 1)
    kk = torch.sum(k.transpose(-2,-1)**2, dim=-1, keepdim=True)  # kk.shape == (B, H, N, 1)
    qk_l2 = qq + inner + kk.transpose(-2, -1)  # qk_l2.shape == (B, H, N, N)
    return qk_l2

def neighbor_mask(pcd, K):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    _, idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    B, N, _ = idx.shape
    mask = torch.zeros(B, N, N, dtype=torch.float32, device=idx.device) # mask.shape == (B, N, N)
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
    output = torch.gather(pcd, 2, idx) # output.shape == (B, C, K)
    return output

def norm_range(x, dim=-1, n_min=0, n_max=1, mode="minmax"):
    if mode == 'minmax':
        x_norm = (x - torch.min(x, dim=dim, keepdim=True)[0])/(torch.max(x, dim=dim, keepdim=True)[0]-torch.min(x, dim=dim, keepdim=True)[0] + 1e-8)
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


def sort_chunk(x, num_bins, dim, descending=False):
    x_sorted, idx_sorted = torch.sort(x, dim=dim, descending=descending)
    x_chunks = torch.chunk(x_sorted, num_bins, dim=dim) # x_chunks.shape == num_bins * (B, H, N/num_bins)
    idx_chunks = torch.chunk(idx_sorted, num_bins, dim=dim) # idx_sorted.shape == num_bins * (B, H, N/num_bins)
    return x_chunks, idx_chunks