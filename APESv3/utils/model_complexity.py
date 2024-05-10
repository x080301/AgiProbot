import wandb
from omegaconf import OmegaConf
from pathlib import Path
from utils import dataloader
from models import cls_model, seg_model
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import hydra
import subprocess
import datetime
import socket
import time
from fvcore.nn import FlopCountAnalysis, parameter_count

from utils.visualization import *
from utils.visualization_data_processing import *
from utils.check_config import set_config_run

from thop import profile


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_flops(model):
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只考虑基本模块，避免重复计算
            module_flops, module_params = profile(module, inputs=(input,), verbose=False)
            print(f"{name}: {module_flops} FLOPs")


def cal_parameters_of_cls(config):
    # hostname = socket.gethostname()

    # config = OmegaConf.load('configs/default.yaml')
    # cmd_config = {
    #     'usr_config': 'configs/token_nonaveragebins_std_cls.yaml',
    #     'datasets': 'modelnet_AnTao420M',
    #     'wandb': {'name': '2024_02_21_01_47_Modelnet_Token_Std'},
    #     'test': {'ddp': {'which_gpu': [0, 1]}}
    # }
    # config = OmegaConf.merge(config, OmegaConf.create(cmd_config))
    #
    # dataset_config=OmegaConf.load(f'configs/datasets/{config.datasets}.yaml')
    # config = OmegaConf.merge(config, dataset_config)

    # check working directory
    # try:
    #     assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    # except:
    #     exit(f'Working directory is not the same as project root. Exit.')

    # get test configurations
    if config.usr_config:
        test_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, test_config)

    # download artifacts
    if config.wandb.enable:
        wandb.login(key=config.wandb.api_key)
        api = wandb.Api()
        artifact = api.artifact(f'{config.wandb.entity}/{config.wandb.project}/{config.wandb.name}:latest')
        if config.test.suffix.enable:
            local_path = f'./artifacts/{config.wandb.name}_{config.test.suffix.remark}'
        else:
            local_path = f'./artifacts/{config.wandb.name}'
        artifact.download(root=local_path)
    else:
        raise ValueError('W&B is not enabled!')

    # overwrite the default config with previous run config
    config.mode = 'test'
    run_config = OmegaConf.load(f'{local_path}/usr_config.yaml')
    if not config.test.suffix.enable:
        config = OmegaConf.merge(config, run_config)
    else:
        OmegaConf.save(config, f'{local_path}/usr_config_test.yaml')
        print(f'Overwrite the previous run config with new run config.')
    config = set_config_run(config, "test")

    if config.datasets.dataset_name == 'modelnet_AnTao420M':
        dataloader.download_modelnet_AnTao420M(config.datasets.url, config.datasets.saved_path)
    elif config.datasets.dataset_name == 'modelnet_Alignment1024':
        dataloader.download_modelnet_Alignment1024(config.datasets.url, config.datasets.saved_path)
    else:
        raise ValueError('Not implemented!')

    # multiprocessing for ddp

    my_model = cls_model.ModelNetModel(config)

    flops, params = profile(my_model, inputs=(torch.randn((1, 3, 2048)),))

    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")

    for name, module in my_model.named_children():

        if name == 'block':
            for name, module in module.named_children():

                if name == 'downsample_list':
                    for name, module in module.named_children():
                        params = count_parameters(module)

                        print(f"downsample layer {name}: {params} parameters")
                elif name == 'feature_learning_layer_list':
                    for name, module in module.named_children():
                        params = count_parameters(module)
                        print(f"feature learning layer {name}: {params} parameters")
                else:
                    params = count_parameters(module)
                    print(f"{name}: {params} parameters")
        else:

            params = count_parameters(module)
            print(f"{name}: {params} parameters")
    print_flops
    # if torch.cuda.is_available():
    #     os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(config.test.ddp.which_gpu).replace(' ', '').replace('[', '').replace(
    #         ']', '')
    #     mp.spawn(test, args=(config,), nprocs=config.test.ddp.nproc_this_node, join=True)
    # else:
    #     raise ValueError('Please use GPU for testing!')


def calculate_model_complexity_cls():
    subprocess.run('nvidia-smi', shell=True, text=True, stdout=None, stderr=subprocess.PIPE)
    config = OmegaConf.load('configs/default.yaml')
    cmd_config = {
        'usr_config': 'configs/boltzmannT01_fix.yaml',
        'datasets': 'modelnet_AnTao420M',
        'wandb': {'name': '2024_04_09_13_39_Modelnet_Token_Std_boltzmann_T0102_norm_sparsesum1_1'},
        'test': {'ddp': {'which_gpu': [3]}}
    }
    config = OmegaConf.merge(config, OmegaConf.create(cmd_config))

    dataset_config = OmegaConf.load(f'configs/datasets/{config.datasets}.yaml')
    dataset_config = OmegaConf.create({'datasets': dataset_config})
    config = OmegaConf.merge(config, dataset_config)

    cal_parameters_of_cls(config)


def cal_parameters_of_seg(config):
    # check working directory
    # try:
    #     assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    # except:
    #     exit(f'Working directory is not the same as project root. Exit.')

    # get test configurations
    if config.usr_config:
        test_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, test_config)

    # download artifacts
    if config.wandb.enable:
        wandb.login(key=config.wandb.api_key)
        api = wandb.Api()
        artifact = api.artifact(f'{config.wandb.entity}/{config.wandb.project}/{config.wandb.name}:latest')
        if config.test.suffix.enable:
            local_path = f'./artifacts/{config.wandb.name}_{config.test.suffix.remark}'
        else:
            local_path = f'./artifacts/{config.wandb.name}'
        artifact.download(root=local_path)
    else:
        raise ValueError('W&B is not enabled!')

    # overwrite the default config with previous run config
    config.mode = 'test'
    run_config = OmegaConf.load(f'{local_path}/usr_config.yaml')
    if not config.test.suffix.enable:
        config = OmegaConf.merge(config, run_config)
    else:
        OmegaConf.save(config, f'{local_path}/usr_config_test.yaml')
        print(f'Overwrite the previous run config with new run config.')
    config = set_config_run(config, "test")

    if config.datasets.dataset_name == 'shapenet_Yi650M':
        dataloader.download_shapenet_Yi650M(config.datasets.url, config.datasets.saved_path)
    elif config.datasets.dataset_name == 'shapenet_AnTao350M':
        dataloader.download_shapenet_AnTao350M(config.datasets.url, config.datasets.saved_path)
    elif config.datasets.dataset_name == 'shapenet_Normal':
        dataloader.download_shapenet_Normal(config.datasets.url, config.datasets.saved_path)
    else:
        raise ValueError('Not implemented!')

    # multiprocessing for ddp
    my_model = seg_model.ShapeNetModel(config)

    flops, params = profile(my_model, inputs=(torch.randn((1, 3, 2048)), torch.randn((1, 16, 1))))

    print(f"Parameters: {params}")
    print(f"FLOPs: {flops}")

    for name, module in my_model.named_children():

        if name == 'block':
            for name, module in module.named_children():

                if name == 'downsample_list':
                    for name, module in module.named_children():
                        params = count_parameters(module)
                        print(f"downsample layer {name}: {params} parameters")
                elif name == 'feature_learning_layer_list':
                    for name, module in module.named_children():
                        params = count_parameters(module)
                        print(f"feature learning layer {name}: {params} parameters")
                elif name == 'upsample_list':
                    for name, module in module.named_children():
                        params = count_parameters(module)
                        print(f"upsample layers {name}: {params} parameters")
                else:
                    if name == 'embedding_list':
                        flops, params = profile(module, inputs=(torch.randn((1, 3, 2048)), torch.randn((1, 16, 1)),))
                        print(f"{name}: {params} parameters, {flops} FLOPs")
                    params = count_parameters(module)
                    print(f"{name}: {params} parameters")
        else:
            params = count_parameters(module)
            print(f"{name}: {params} parameters")

    # if torch.cuda.is_available():
    #     os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(config.test.ddp.which_gpu).replace(' ', '').replace('[', '').replace(
    #         ']', '')
    #     mp.spawn(test, args=(config,), nprocs=config.test.ddp.nproc_this_node, join=True)
    # else:
    #     raise ValueError('Please use GPU for testing!')


def calculate_model_complexity_seg():
    subprocess.run('nvidia-smi', shell=True, text=True, stdout=None, stderr=subprocess.PIPE)
    config = OmegaConf.load('configs/default.yaml')
    cmd_config = {
        'usr_config': 'configs/token_nonaveragebins_std_seg_logmean.yaml',
        'datasets': 'shapenet_AnTao350M',
        'wandb': {'name': '2024_04_16_01_26_Shapenet_Token_Std_logmean_1'},
        'test': {'ddp': {'which_gpu': [0, 1]}}
    }
    config = OmegaConf.merge(config, OmegaConf.create(cmd_config))

    dataset_config = OmegaConf.load(f'configs/datasets/{config.datasets}.yaml')
    dataset_config = OmegaConf.create({'datasets': dataset_config})
    config = OmegaConf.merge(config, dataset_config)

    cal_parameters_of_seg(config)
