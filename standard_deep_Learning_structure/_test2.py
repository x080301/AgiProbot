import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # 生成随机点云数据
# num_points = 1000
# x = np.random.normal(size=num_points)
# y = np.random.normal(size=num_points)
# z = np.random.normal(size=num_points)


# def draw_3dhisgram(x):

data3d = np.random.normal(size=(1000, 3))
def draw_3d_histogram(data3d,num_bins):

# 将点云数据分成立方体
num_bins = 10
hist, edges = np.histogramdd(data3d, bins=num_bins)

# 获取每个立方体的中心点
x_centers = (edges[0][1:] + edges[0][:-1]) / 2
y_centers = (edges[1][1:] + edges[1][:-1]) / 2
z_centers = (edges[2][1:] + edges[2][:-1]) / 2

# 创建一个三维坐标轴对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 在每个中心点绘制一个球形点云，大小为每个立方体内的点数
for i in range(num_bins):
    for j in range(num_bins):
        for k in range(num_bins):
            center = (x_centers[i], y_centers[j], z_centers[k])
            size = hist[i, j, k]
            ax.scatter(*center, s=size, c='blue')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
