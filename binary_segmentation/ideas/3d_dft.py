import datetime
import os
import platform
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data_preprocess.data_loader import MotorDataset, MotorDatasetTest
from model.pct import PCT_semseg
from utilities import util
from utilities.config import get_parser
from utilities.lr_scheduler import CosineAnnealingWithWarmupLR
import matplotlib.pyplot as plt


def find_neighbor_in_d(point_cloud, d, output_size):
    """

    :param point_cloud: input numpy array (B,C,N)
    :param d: maximun distance of the neighbor points
    :param output_size: size of output
    :return: numpy array (B,output_size,output_size,output_size,N)
    """

    inner = -2 * torch.matmul(point_cloud.transpose(2, 1), point_cloud)
    xx = torch.sum(point_cloud ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B,N,N)

    b, n1, n2 = np.where(pairwise_distance < d)

    output = point_cloud[b, :, n1]


class BinarySegmentation:
    # 'config/binary_segmentation.yaml' should be at the end. It can be changed latter.
    files_to_save = ['train.py', 'model/pct.py', 'data_preprocess/data_loader.py', 'config/binary_segmentation.yaml']

    def __init__(self, config_dir='config/binary_segmentation.yaml'):

        # ******************* #
        # load arguments
        # ******************* #
        self.config_dir = config_dir
        self.args = get_parser(config_dir=self.config_dir)
        self.device = torch.device("cuda")

        # ******************* #
        # local or server?
        # ******************* #
        system_type = platform.system().lower()  # 'windows' or 'linux'
        self.is_local = True if system_type == 'windows' else False
        if self.is_local:
            self.args.npoints = 1024
            self.args.sample_rate = 1.
            self.args.ddp.gpus = 1

    def init_training(self):

        # ******************* #
        # load data set
        # ******************* #
        if self.is_local:
            data_set_direction = 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_4debug'  # 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_npy'
        else:
            data_set_direction = self.args.data_dir

        print("start loading training data ...")
        train_dataset = MotorDataset(mode='train',
                                     data_dir=data_set_direction,
                                     num_class=self.args.num_segmentation_type, num_points=self.args.npoints,  # 4096
                                     test_area='Validation', sample_rate=self.args.sample_rate)

        para_workers = 0 if self.is_local else 8
        self.train_loader = DataLoader(train_dataset, num_workers=para_workers, batch_size=self.args.train_batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       worker_init_fn=lambda x: np.random.seed(x + int(time.time())))

    def train(self):

        self.init_training()

        num_points_in_d = None
        print("------------------------------------------")
        for i, (points, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), smoothing=0.9):
            # ******************* #
            # forwards
            # ******************* #
            points, target = points.to(self.device), target.to(self.device)
            points = util.normalize_data(points)

            points, _ = util.rotate_per_batch(points, None)

            points = points.permute(0, 2, 1).float()

            inner = -2 * torch.matmul(points.transpose(2, 1), points)
            xx = torch.sum(points ** 2, dim=1, keepdim=True)
            distances = xx + inner + xx.transpose(2, 1)  # (B,N,N)
            # print(distances)

            # distances = np.asarray(distances)
            count = torch.sum(distances <= 0.004, axis=-1)

            if num_points_in_d is None:
                num_points_in_d = np.asarray(count.cpu()).flatten()
            else:
                num_points_in_d = np.append(num_points_in_d, np.asarray(count.cpu()).flatten())
            # seg_pred, trans = self.model(points.float())  # _  (B,3,N) -> (B,segment_type,N),(B,3,3)

        mean = np.mean(num_points_in_d)
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')

        plt.hist(num_points_in_d, bins=range(np.max(num_points_in_d) + 2))
        plt.xlabel('Number of points within distance d')
        plt.ylabel('Frequency')
        plt.title('Histogram of points within distance d')
        plt.show()


if __name__ == "__main__":
    binarysegmentation = BinarySegmentation(config_dir='../config/binary_segmentation.yaml')
    binarysegmentation.train()
