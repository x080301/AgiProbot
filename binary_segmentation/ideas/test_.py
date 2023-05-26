import cv2
import numpy as np


def get_gaussian_kernel(kernel_size=3, sigma=1):
    # 使用cv2.getGaussianKernel生成高斯卷积核
    kernel_1_dimension = cv2.getGaussianKernel(kernel_size, sigma).flatten()
    # print(type(kernel))
    # print(kernel.shape)
    # print(kernel_1_dimension)
    kernel_2_dimension = np.outer(kernel_1_dimension, kernel_1_dimension)  # 将一维核扩展为二维
    # print(kernel_2_dimension)

    # 将二维张量重复拓展为三维
    kernel_3_dimension = np.expand_dims(kernel_2_dimension, axis=0)  # 在第0维度添加一个维度
    kernel_3_dimension = np.repeat(kernel_3_dimension, 3, axis=0)  # 沿着第0维度进行重复

    # 在通道维度上分别乘上 x、y、z
    kernel_3_dimension = kernel_3_dimension * kernel_1_dimension[:, np.newaxis, np.newaxis]

    # 打印结果
    # print(kernel_3_dimension)
    return kernel_3_dimension


print(get_gaussian_kernel())
