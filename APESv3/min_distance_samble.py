import pickle
import torch
import os


def sum_of_min_distance(pc_a, pc_b, no_self):
    min_distance = pc_a - torch.permute(pc_b, (1, 0, 2))
    min_distance = torch.sum(min_distance ** 2, dim=2)

    if no_self:
        min_distance[min_distance == 0] += 100

    min_distance, _ = torch.min(min_distance, dim=1)
    return torch.sum(min_distance)


min_distance_2048 = 0
min_distance_1024 = 0
min_distance_512 = 0
min_distance_256 = 0
min_distance_128 = 0
min_distance_64 = 0
min_distance_32 = 0
min_distance_16 = 0

file_dir = 'E:/datasets/APES/test_results/2024_02_21_01_47_Modelnet_Token_Std/'
counter = 0
for file_name in os.listdir(file_dir):
    if '.pkl' in file_name:
        print(f'loading: {os.path.join(file_dir, file_name)}')
        with open(os.path.join(file_dir, file_name), 'rb') as f:
            data = pickle.load(f)
        counter += 1
    else:
        continue

    for point_cloud_index in range(16):
        pc_2048 = data['samples'][point_cloud_index, :, :]

        pc_1024 = pc_2048[data['idx_down'][point_cloud_index][0].flatten(), :].reshape((-1, 1, 3))
        pc_512 = pc_2048[data['idx_down'][point_cloud_index][0].flatten()[:512], :].reshape((-1, 1, 3))
        pc_256 = pc_2048[data['idx_down'][point_cloud_index][0].flatten()[:256], :].reshape((-1, 1, 3))
        pc_128 = pc_2048[data['idx_down'][point_cloud_index][0].flatten()[:128], :].reshape((-1, 1, 3))
        pc_64 = pc_2048[data['idx_down'][point_cloud_index][0].flatten()[:64], :].reshape((-1, 1, 3))
        pc_32 = pc_2048[data['idx_down'][point_cloud_index][0].flatten()[:32], :].reshape((-1, 1, 3))
        pc_16 = pc_2048[data['idx_down'][point_cloud_index][0].flatten()[:16], :].reshape((-1, 1, 3))

        pc_2048 = pc_2048.reshape((-1, 1, 3))

        min_distance_2048 += sum_of_min_distance(pc_2048, pc_2048, True)
        min_distance_1024 += sum_of_min_distance(pc_2048, pc_1024, True)
        min_distance_512 += sum_of_min_distance(pc_2048, pc_512, True)
        min_distance_256 += sum_of_min_distance(pc_2048, pc_256, True)
        min_distance_128 += sum_of_min_distance(pc_2048, pc_128, True)
        min_distance_64 += sum_of_min_distance(pc_2048, pc_64, True)
        min_distance_32 += sum_of_min_distance(pc_2048, pc_32, True)
        min_distance_16 += sum_of_min_distance(pc_2048, pc_16, True)

min_distance_2048 /= (2048 * counter * 16)
min_distance_1024 /= (2048 * counter * 16)
min_distance_512 /= (2048 * counter * 16)
min_distance_256 /= (2048 * counter * 16)
min_distance_128 /= (2048 * counter * 16)
min_distance_64 /= (2048 * counter * 16)
min_distance_32 /= (2048 * counter * 16)
min_distance_16 /= (2048 * counter * 16)

print(
    f'{min_distance_2048}\t{min_distance_1024}\t{min_distance_512}\t{min_distance_256}\t{min_distance_128}\t{min_distance_64}\t{min_distance_32}\t{min_distance_16}')
