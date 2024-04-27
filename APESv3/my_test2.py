import torch

import pickle
import torch
import matplotlib.pyplot as plt

a = torch.load(r'C:\Users\Lenovo\Desktop\a.pt').cpu()
b = torch.load(r'C:\Users\Lenovo\Desktop\a.pt').cpu()

print(a.shape)

diff = torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=2)  # diff.shape == (B,N, M, C)
pairwise_distance = torch.sum(diff ** 2, dim=-1)


# a_mean = torch.mean(a, dim=1, keepdim=True)
# a = a - a_mean
# b = b - a_mean
#
# a_std = torch.mean(torch.std(a, dim=1, keepdim=True), dim=2, keepdim=True)
# a = a / a_std
# b = b / a_std
#
# inner = 2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
# aa = torch.sum(a ** 2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
# bb = torch.sum(b ** 2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
# pairwise_distance = aa - inner + bb.transpose(2, 1)
# # pairwise_distance=torch.diagonal(pairwise_distance, dim1=1, dim2=2)

tensor = pairwise_distance.flatten()#.cpu()
hist_values, bin_edges = torch.histogram(tensor, bins=200)#,range=(-5e-11,7e-11))#,range=(0,7e-11))#,range=(0,2.5e-9))  # ,range=(0,1e-11))#, range=(-2.45, 5))

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist_values, width=(bin_edges[1] - bin_edges[0]))  # , color='lightsteelblue')
# plt.title('Histogram of sampling scores with z-score normalization')
# plt.xlabel('Value of Pairwise_distance')
plt.ylabel('Frequency')
plt.yscale('log')
plt.grid(True)
plt.show()
