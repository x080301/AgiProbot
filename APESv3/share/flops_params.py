from models import comp_model
import torch
from ptflops import get_model_complexity_info


from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
import os
from models import cls_model
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.visualization import *
from utils.visualization_data_processing import *
from utils.check_config import set_config_run


@hydra.main(version_base=None, config_path="./configs", config_name="default.yaml")
def main(config):

    # check working directory
    try:
        assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    except:
        exit(f'Working directory is not the same as project root. Exit.')

    # get test configurations
    if config.usr_config:
        test_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, test_config)

    # multiprocessing for ddp
    if torch.cuda.is_available():
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.test.ddp.which_gpu).replace(' ', '').replace('[', '').replace(']', '')
        mp.spawn(test, args=(config,), nprocs=config.test.ddp.nproc_this_node, join=True)
    else:
        raise ValueError('Please use GPU for testing!')


def test(local_rank, config):

    rank = config.test.ddp.rank_starts_from + local_rank
    # set files path
    
    # process initialization
    os.environ['MASTER_ADDR'] = str(config.test.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(config.test.ddp.master_port)
    os.environ['WORLD_SIZE'] = str(config.test.ddp.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # gpu setting
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)  # which gpu is used by current process

    # get model
    my_model = cls_model.ModelNetModel(config)
    my_model.eval()
    my_model = my_model.to(device)
    flops, params = get_model_complexity_info(my_model, (3, 2048), as_strings=True, print_per_layer_stat=True)

if __name__ == '__main__':
    main()