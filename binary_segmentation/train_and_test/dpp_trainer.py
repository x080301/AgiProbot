import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


class BinarySegmentationDPP:
    files_to_save = ['config', 'data_preprocess', 'ideas', 'model', 'train_and_test', 'train_line', 'utilities',
                     'train.py', 'train_line.py']

    def __init__(self, config_dir='config/binary_segmentation.yaml'):
        '''# ******************* #
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

        # ******************* #
        # load ML model
        # ******************* #
        self.model = PCTSeg(self.args).to(self.device)
        self.model = nn.DataParallel(self.model)'''
        print("use", torch.cuda.device_count(), "GPUs for training")

    def example(self, rank, world_size):
        # 初始化
        torch.manual_seed(100)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        # 创建模型
        model = nn.Linear(10, 10).cuda(rank)
        # 放入DDP
        ddp_model = DDP(model, device_ids=[rank])
        ddp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ddp_model)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        # 进行前向后向计算
        for i in range(10):
            outputs = ddp_model(torch.randn(20, 10).to(rank))
            labels = torch.randn(20, 10).to(rank)
            loss_fn(outputs, labels).backward()
            optimizer.step()
        print(rank)

    def dpp_train(self):
        world_size = torch.cuda.device_count()
        mp.spawn(self.example,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)


if __name__ == "__main__":
    binarySegmentation_dpp = BinarySegmentationDPP('config/binary_segmentation.yaml')
    binarySegmentation_dpp.dpp_train()