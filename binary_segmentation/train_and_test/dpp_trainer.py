import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import platform

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
from utilities.config import get_parser
from model.pct import PCTSeg


class BinarySegmentationDPP:
    def __init__(self, config_dir='config/binary_segmentation.yaml'):
        # ******************* #
        # load arguments
        # ******************* #
        self.config_dir = config_dir
        self.args = get_parser(config_dir=self.config_dir)
        print("use", torch.cuda.device_count(), "GPUs for training")

        # ******************* #
        # local or server?
        # ******************* #
        system_type = platform.system().lower()  # 'windows' or 'linux'
        self.is_local = True if system_type == 'windows' else False
        if self.is_local:
            self.args.npoints = 1024
            self.args.sample_rate = 1.
            self.args.ddp.gpus = 1

    def example(self, rank, world_size):
        torch.manual_seed(0)
        # 初始化
        backend = 'gloo' if self.is_local else 'nccl'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
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

    def main(self):
        world_size = torch.cuda.device_count()
        mp.spawn(self.example,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)


if __name__ == "__main__":
    main()
