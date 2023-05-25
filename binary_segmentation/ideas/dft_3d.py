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


def find_neighbor_in_d(point_cloud, d_square=15, output_size=11, fft=True):
    """

    :param point_cloud: input numpy array (B,C,N)
    :param d_square: maximun distance of the neighbor points
    :param output_size: size of output
    :param fft: True fft before output, False no fft
    :return: numpy array (B,output_size,output_size,output_size,N)
    """

    # calculate distance

    inner = -2 * torch.matmul(point_cloud.transpose(2, 1), point_cloud)
    xx = torch.sum(point_cloud ** 2, dim=1, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(2, 1)  # (B,N,N)

    # find neighbors with distance < d_square
    b, n1, n2 = torch.nonzero(pairwise_distance < d_square, as_tuple=True)
    '''
    pairwise_distance = np.asarray(pairwise_distance.cpu())
    b, n1, n2 = np.where(pairwise_distance < d_square)
    '''

    neighbors = point_cloud[b, :, n1]
    itself = point_cloud[b, :, n2]
    related = neighbors - itself

    related_x = related[:, 0].contiguous()
    related_y = related[:, 1].contiguous()
    related_z = related[:, 2].contiguous()

    # digitize
    bins = torch.tensor(range(output_size - 1)).cuda() + 1  # 1~outputsize) (0,outputsize0
    bins = bins / output_size  # 1/outputsize~1 (0,1)

    d = d_square ** 0.5
    bins = bins * d - d / 2.0  # (-d/2,d/2)

    related_x_digit = torch.bucketize(related_x, bins, right=False)
    related_y_digit = torch.bucketize(related_y, bins, right=False)
    related_z_digit = torch.bucketize(related_z, bins, right=False)

    # prepare output
    points_with_neighbors = torch.zeros(
        (point_cloud.shape[0], output_size, output_size, output_size, point_cloud.shape[2])).cuda()
    points_with_neighbors[b, related_x_digit, related_y_digit, related_z_digit, n2] = 1

    if fft:
        points_with_neighbors_spectrum = torch.fft.fftn(points_with_neighbors, dim=(1, 2, 3))
        points_with_neighbors_spectrum = torch.fft.fftshift(points_with_neighbors_spectrum, dim=(1, 2, 3))

        points_with_neighbors_spectrum = 20 * torch.log10(torch.abs(points_with_neighbors_spectrum))
        # points_with_neighbors_spectrum = 20 * torch.log10(torch.abs(points_with_neighbors_spectrum))
        return points_with_neighbors_spectrum
    else:
        return points_with_neighbors


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

    def count_point_num(self):

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

            points = points.permute(0, 2, 1).float()  # _       (B,N,3) -> (B,3,N)

            print(find_neighbor_in_d(points, 15, 11, fft=True).shape)


if __name__ == "__main__":
    binarysegmentation = BinarySegmentation(config_dir='../config/binary_segmentation.yaml')
    binarysegmentation.train()
