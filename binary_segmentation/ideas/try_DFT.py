import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

'''
image = Image.open('C:/Users/Lenovo/Desktop/lena.jpg')
image=image.rotate(90)
image=np.asarray(image)

# 读取图像
#image = plt.imread('C:/Users/Lenovo/Desktop/lena.jpg')  # 替换为你的图像路径
image=image/255
print(type(image))
print(image.shape)
'''


def create_matrix(x, y):
    matrix = np.zeros((50, 50))
    matrix[x, y] = 1
    return matrix


image = np.zeros((50, 50))

for i in range(20):
    image += create_matrix(random.randint(0, 49), random.randint(0, 49))

# 执行二维傅里叶变换
fft_image = np.fft.fft2(image)

# 将零频率分量移动到图像中心
fft_shifted = np.fft.fftshift(fft_image)

# 计算幅度谱（取对数以增强显示效果）
magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))

# 绘制原始图像和幅度谱
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.show()
