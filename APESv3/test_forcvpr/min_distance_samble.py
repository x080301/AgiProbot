import pickle
import torch
import os

with open('E:/datasets/APES/test_results/2024_02_21_01_47_Modelnet_Token_Std/intermediate_result_0.pkl', 'rb') as file:
    data = pickle.load(file)

print(data.keys())
print(data['samples'].shape)
print(len(data['idx_down']))
print(data['idx_down'][0][0].shape)
print(torch.max(data['idx_down'][0][1]))
print(torch.max(data['idx_down'][0][0]))
print(torch.max(data['samples']))
print(torch.min(data['samples']))
#
distance_2048_2048 = 0
distance_2048_1024 = 0
distance_2048_512 = 0
distance_2048_1024_no_self = 0
distance_2048_512_no_self = 0
distance_1024_1024 = 0
distance_1024_512 = 0
distance_1024_512_no_self = 0
distance_512_512 = 0

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

        pc_1024_index = data['idx_down'][point_cloud_index][0].flatten()
        pc_1024 = pc_2048[pc_1024_index, :]

        pc_512_index = data['idx_down'][point_cloud_index][1].flatten()
        pc_512 = pc_1024[pc_512_index, :]

        pc_2048 = torch.reshape(pc_2048, (2048, 1, 3))
        pc_1024 = torch.reshape(pc_1024, (1024, 1, 3))
        pc_512 = torch.reshape(pc_512, (512, 1, 3))

        min_distance = pc_2048 - torch.permute(pc_2048, (1, 0, 2))
        min_distance = torch.sum(min_distance ** 2, dim=2)
        min_distance[min_distance == 0] += 100
        min_distance, _ = torch.min(min_distance, dim=1)
        distance_2048_2048 += torch.sum(min_distance)

        min_distance = pc_2048 - torch.permute(pc_1024, (1, 0, 2))
        min_distance = torch.sum(min_distance ** 2, dim=2)
        # min_distance[min_distance == 0] += 100
        min_distance, _ = torch.min(min_distance, dim=1)
        distance_2048_1024 += torch.sum(min_distance)

        min_distance = pc_2048 - torch.permute(pc_1024, (1, 0, 2))
        min_distance = torch.sum(min_distance ** 2, dim=2)
        min_distance[min_distance == 0] += 100
        min_distance, _ = torch.min(min_distance, dim=1)
        distance_2048_1024_no_self += torch.sum(min_distance)

        min_distance = pc_2048 - torch.permute(pc_512, (1, 0, 2))
        min_distance = torch.sum(min_distance ** 2, dim=2)
        # min_distance[min_distance == 0] += 100
        min_distance, _ = torch.min(min_distance, dim=1)
        distance_2048_512 += torch.sum(min_distance)

        min_distance = pc_2048 - torch.permute(pc_512, (1, 0, 2))
        min_distance = torch.sum(min_distance ** 2, dim=2)
        min_distance[min_distance == 0] += 100
        min_distance, _ = torch.min(min_distance, dim=1)
        distance_2048_512_no_self += torch.sum(min_distance)

        min_distance = pc_1024 - torch.permute(pc_1024, (1, 0, 2))
        min_distance = torch.sum(min_distance ** 2, dim=2)
        min_distance[min_distance == 0] += 100
        min_distance, _ = torch.min(min_distance, dim=1)
        distance_1024_1024 += torch.sum(min_distance)

        min_distance = pc_1024 - torch.permute(pc_512, (1, 0, 2))
        min_distance = torch.sum(min_distance ** 2, dim=2)
        min_distance[min_distance == 0] += 100
        min_distance, _ = torch.min(min_distance, dim=1)
        distance_1024_512 += torch.sum(min_distance)

        min_distance = pc_1024 - torch.permute(pc_512, (1, 0, 2))
        min_distance = torch.sum(min_distance ** 2, dim=2)
        # min_distance[min_distance == 0] += 100
        min_distance, _ = torch.min(min_distance, dim=1)
        distance_1024_512_no_self += torch.sum(min_distance)

        min_distance = pc_512 - torch.permute(pc_512, (1, 0, 2))
        min_distance = torch.sum(min_distance ** 2, dim=2)
        min_distance[min_distance == 0] += 100
        min_distance, _ = torch.min(min_distance, dim=1)
        distance_512_512 += torch.sum(min_distance)

        # for point_index in range(2048):
        #     min_distance = torch.sum((pc_2048 - pc_2048[point_index, :]) ** 2, dim=1)
        #     min_distance = torch.masked_select(min_distance, min_distance != 0)
        #     distance_2048_2048 += torch.min(min_distance)
        #
        #     min_distance = torch.sum((pc_1024 - pc_2048[point_index, :]) ** 2, dim=1)
        #     # min_distance = torch.masked_select(min_distance, min_distance != 0)
        #     distance_2048_1024 += torch.min(min_distance)
        #
        #     min_distance = torch.sum((pc_512 - pc_2048[point_index, :]) ** 2, dim=1)
        #     # min_distance = torch.masked_select(min_distance, min_distance != 0)
        #     distance_2048_512 += torch.min(min_distance)
        #
        #     min_distance = torch.sum((pc_1024 - pc_2048[point_index, :]) ** 2, dim=1)
        #     min_distance = torch.masked_select(min_distance, min_distance != 0)
        #     distance_2048_1024_no_self += torch.min(min_distance)
        #
        #     min_distance = torch.sum((pc_512 - pc_2048[point_index, :]) ** 2, dim=1)
        #     min_distance = torch.masked_select(min_distance, min_distance != 0)
        #     distance_2048_512_no_self += torch.min(min_distance)
        #
        # for point_index in range(1024):
        #     min_distance = torch.sum((pc_1024 - pc_1024[point_index, :]) ** 2, dim=1)
        #     min_distance = torch.masked_select(min_distance, min_distance != 0)
        #     distance_1024_1024 += torch.min(min_distance)
        #
        #     min_distance = torch.sum((pc_512 - pc_1024[point_index, :]) ** 2, dim=1)
        #     # min_distance = torch.masked_select(min_distance, min_distance != 0)
        #     distance_1024_512 += torch.min(min_distance)
        #
        #     min_distance = torch.sum((pc_512 - pc_1024[point_index, :]) ** 2, dim=1)
        #     min_distance = torch.masked_select(min_distance, min_distance != 0)
        #     distance_1024_512_no_self += torch.min(min_distance)
        #
        # for point_index in range(512):
        #     min_distance = torch.sum((pc_512 - pc_512[point_index, :]) ** 2, dim=1)
        #     min_distance = torch.masked_select(min_distance, min_distance != 0)
        #     distance_512_512 += torch.min(min_distance)

print(f'{counter} batches in total')
print(f'distance_2048_2048 / 2048 / 16={distance_2048_2048 / 2048 / 16 / counter}')
print(f'distance_2048_1024 / 2048 / 16={distance_2048_1024 / 2048 / 16 / counter}')
print(f'distance_2048_512 / 2048 / 16={distance_2048_512 / 2048 / 16 / counter}')
print(f'distance_2048_1024_no_self / 2048 / 16={distance_2048_1024_no_self / 2048 / 16 / counter}')
print(f'distance_2048_512_no_self / 2048 / 16={distance_2048_512_no_self / 2048 / 16 / counter}')
print(f'distance_1024_1024 / 1024 / 16={distance_1024_1024 / 1024 / 16 / counter}')
print(f'distance_1024_512 / 1024 / 16={distance_1024_512 / 1024 / 16 / counter}')
print(f'distance_1024_512_no_self / 1024 / 16={distance_1024_512_no_self / 1024 / 16 / counter}')
print(f'distance_512_512 / 512 / 16={distance_512_512 / 512 / 16 / counter}')
