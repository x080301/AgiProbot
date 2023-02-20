from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os


class MotorDataset(Dataset):
    def __init__(self, data_pandas_csv, mode, num_points=4096):
        self.data = data_pandas_csv
        self.mode = mode
        self.num_points = num_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # load points(N*3) and segementation_label(N) from .np file
        # the order of the array should be shuffled and the array should be resampled to num_points(4096 default) points
        point_cloud_dir = self.data.iloc[index, 0]

        data_from_file = np.load(point_cloud_dir)
        point_cloud = data_from_file[:, 0:3]
        segementation_label = data_from_file[:, 6]

        choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)
        point_cloud = point_cloud[choice, :]
        segementation_label = segementation_label[choice]

        # TODO
        # goals?何意

        return point_cloud, segementation_label
