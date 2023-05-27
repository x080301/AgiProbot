import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import platform
import time
import datetime
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utilities.config import get_parser
from model.pct import PCTSeg
from data_preprocess.data_loader import MotorDataset

class BinarySegmentationDPP:
    files_to_save = ['config', 'data_preprocess', 'ideas', 'model', 'train_and_test', 'train_line', 'utilities',
                     'train.py', 'train_line.py']

    def __init__(self, config_dir='config/binary_segmentation.yaml'):
        # ******************* #
        # load arguments
        # ******************* #
        self.config_dir = config_dir
        self.args = get_parser(config_dir=self.config_dir)
        print("use", torch.cuda.device_count(), "GPUs for training")

        if self.args.random_seed == 0:
            self.random_seed = int(time.time())
        else:
            self.random_seed = self.args.train.random_seed

        # ******************* #
        # local or server?
        # ******************* #
        system_type = platform.system().lower()  # 'windows' or 'linux'
        self.is_local = True if system_type == 'windows' else False
        if self.is_local:
            self.args.npoints = 1024
            self.args.sample_rate = 1.
            self.args.ddp.gpus = 1
            self.data_set_direction = 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_4debug'
            # 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_npy'
        else:
            self.data_set_direction = self.args.data_dir

        # ******************* #
        # make directions
        # ******************* #
        if self.is_local:
            direction = 'outputs/' + self.args.titel + '_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        else:
            direction = '/data/users/fu/' + self.args.titel + '_outputs/' + \
                        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        if not os.path.exists(direction + '/checkpoints'):
            os.makedirs(direction + '/checkpoints')
        if not os.path.exists(direction + '/train_log'):
            os.makedirs(direction + '/train_log')
        if not os.path.exists(direction + '/tensorboard_log'):
            os.makedirs(direction + '/tensorboard_log')
        self.checkpoints_direction = direction + '/checkpoints/'

        # ******************* #
        # save mode and parameters
        # ******************* #
        for file_name in self.files_to_save:
            if '.' in file_name:
                shutil.copyfile(file_name, direction + '/train_log/' + file_name.split('/')[-1])
            else:
                shutil.copytree(file_name, direction + '/train_log/' + file_name.split('/')[-1])

        with open(direction + '/train_log/' + 'random_seed_' + str(self.random_seed) + '.txt', 'w') as f:
            f.write('')

        self.save_direction = direction

        # ******************* #
        # load data set
        # ******************* #
        print("start loading training data ...")

        train_dataset = MotorDataset(mode='train',
                                     data_dir=self.data_set_direction,
                                     num_class=self.args.num_segmentation_type, num_points=self.args.npoints,  # 4096
                                     test_area='Validation', sample_rate=self.args.sample_rate)
        '''self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                             num_replicas=self.args.ddp.world_size,
                                                                             rank=rank
                                                                             )'''
        print("start loading test data ...")
        valid_dataset = MotorDataset(mode='valid',
                                     data_dir=self.data_set_direction,
                                     num_class=self.args.num_segmentation_type, num_points=self.args.npoints,  # 4096
                                     test_area='Validation', sample_rate=1.0)
        '''self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                                             num_replicas=self.args.ddp.world_size,
                                                                             rank=rank
                                                                             )'''

        '''# para_workers = 0 if self.is_local else 8
        self.train_loader = DataLoader(train_dataset,
                                       # num_workers=para_workers,
                                       batch_size=self.args.train_batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       # worker_init_fn=lambda x: np.random.seed(x + int(time.time())),  # TODO 是否有影响？
                                       pin_memory=True,
                                       sampler=self.train_sampler
                                       )
        self.num_train_batch = len(self.train_loader)

        self.validation_loader = DataLoader(valid_dataset,
                                            # num_workers=para_workers,
                                            pin_memory=True,
                                            sampler=self.valid_sampler,
                                            batch_size=self.args.test_batch_size,
                                            shuffle=True,
                                            drop_last=True
                                            )
        self.num_valid_batch = len(self.validation_loader)'''

        # ******************* #
        # dpp
        # ******************* #
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    def train(self, rank, world_size):
        torch.manual_seed(0)
        # 初始化
        backend = 'gloo' if self.is_local else 'nccl'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if rank == 0:
            self.log_writer = SummaryWriter(self.save_direction + '/tensorboard_log')

        # 创建模型
        model = nn.Linear(10, 10).to(rank)
        # 放入DDP
        ddp_model = DDP(model, device_ids=[rank])
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        # 进行前向后向计算
        print(rank)
        for i in range(10):
            outputs = ddp_model(torch.randn(20, 10).to(rank))
            labels = torch.randn(20, 10).to(rank)
            loss_fn(outputs, labels).backward()
            optimizer.step()

    def train_dpp(self):
        world_size = torch.cuda.device_count()
        mp.spawn(self.train,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)


if __name__ == "__main__":
    main()
