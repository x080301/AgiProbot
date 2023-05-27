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
    def __init__(self):
        pass

    def example(self,rank, world_size):
        # 初始化
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
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
