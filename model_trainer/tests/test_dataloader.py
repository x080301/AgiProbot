import unittest
from data_preprocess.data_loader import MotorDataset
from torch.utils.data import DataLoader
import numpy as np
import time
import torch
from tqdm import tqdm

print("start loading training data ...")
train_dataset = MotorDataset(split='train',
                             data_root='E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_npy',
                             num_class=2, num_points=1024,  # 4096
                             test_area='Validation', sample_rate=1.0)
print("start loading test data ...")
validation_set = MotorDataset(split='test',
                              data_root='E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_npy',
                              num_class=2, num_points=1024,  # 4096
                              test_area='Validation', sample_rate=1.0)
train_loader = DataLoader(train_dataset, num_workers=0, batch_size=4, shuffle=True,
                          drop_last=True,
                          worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
validation_loader = DataLoader(validation_set, num_workers=0, batch_size=4, shuffle=True,
                               drop_last=False)

device = torch.device("cuda")
# Try to load models

weights = torch.Tensor(train_dataset.labelweights).cuda()
persentige = torch.Tensor(train_dataset.persentage).cuda()

scale = weights * persentige
scale = 1 / scale.sum()
weights *= scale

len_train = len(train_loader)
len_valid = len(validation_loader)

single_batch_data = None
for epoch in range(2):
    ####################
    # Train
    ####################
    for i, (points, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):

        if single_batch_data is None:
            single_batch_data = [points, target]
        points, target = points.to(device), target.to(device)

    ####################
    # Validation
    ####################
    with torch.no_grad():

        for i, (points, target) in tqdm(enumerate(validation_loader), total=len(validation_loader),
                                        smoothing=0.9):
            points, target = points.to(device), target.to(device)

init_flag = True


class TestMotorDataset(unittest.TestCase):

    def setUp(self):
        super().setUp()

    # @unittest.skip('pass')
    def test_pipline_01(self):
        self.assertTrue(init_flag)

    def test_total_data_num_02(self):
        print(len_train)
        print(len_valid)

    def test_one_batch_data_03(self):
        print(type(single_batch_data[0]))
        print(single_batch_data[0].shape)
        print(single_batch_data[1].shape)
        print(single_batch_data[0][:, 0:-1:200, :])
        print(single_batch_data[1][:, 0:-1:200])

    def test_weights_04(self):
        print(type(weights))
        print(weights.shape)
        print(weights)

    def tearDown(self):
        super().tearDown()
