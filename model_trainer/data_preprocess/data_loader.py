from torch.utils.data import Dataset
import torch

import numpy as np


def data_preloading(data_pandas_csv):
    """
    data_preloading(data_pandas_csv)
    read pcd data to make a csv file, which consists data direction and sampled index

    Args:
        data_pandas_csv (str): csv_direction

    Returns:
        pandas.core.frame.DataFrame
    """
    pass


class MotorDataset(Dataset):
    def __init__(self, data_pandas_csv, mode, num_points=4096):
        """
        MotorDataset(data_pandas_csv, mode, num_points=4096)
        Dataloader

        Args:
            data_pandas_csv (str or pandas.core.frame.DataFrame): str, make csv file before taining;

        Returns:
            None
        """

        self.mode = mode
        self.num_points = num_points
        self.classification_label_list = ['A0', 'A1', 'A2', 'B0']

        if type(data_pandas_csv) is str:
            self.data = data_preloading(data_pandas_csv)
        else:
            self.data = data_pandas_csv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        point_cloud_dir = self.data.iloc[index, 0]

        # ***********************************************************
        # load points(N*3) and segmentation_label(N) from .np file
        # the order of the array should be shuffled and the array should be resampled to num_points(4096 default) points

        data_from_file = np.load(point_cloud_dir)
        point_cloud = data_from_file[:, 0:3]
        segmentation_label = data_from_file[:, 6]

        choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)
        point_cloud = point_cloud[choice, :]
        segmentation_label = segmentation_label[choice]
        # ***********************************************************

        # ***********************************************************
        # get classification label(motor type) from the file name

        classification_label = 4
        for i, (label) in self.classification_label_list:
            if label in data_from_file:
                classification_label = i
        # ***********************************************************

        # ***********************************************************

        # TODO
        # goals?何意

        return point_cloud, segmentation_label, classification_label

if __name__ == "__main__":
    pass