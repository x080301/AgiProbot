from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode

        if mode == 'train':
            self.transform = tv.transforms.Compose(
                [tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])
        else:  # they can be different.
            self.transform = tv.transforms.Compose(
                [tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        #
        image_dir = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1:]
        label = torch.tensor(label, dtype=torch.float32)
        #
        image_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], image_dir)
        image = imread(image_dir)
        #
        image = gray2rgb(image)
        image = self.transform(image)

        return image, label
