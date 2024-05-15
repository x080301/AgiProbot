import wandb
import hydra
import subprocess
import socket
import sys
from omegaconf import OmegaConf
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist
import time
from tqdm import tqdm

from utils import dataloader
from models import seg_model_inference_time
from utils.visualization_data_processing import *
from utils.check_config import set_config_run


def main_without_Decorators(config):
    # check working directory
    try:
        assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    except:
        exit(f'Working directory is not the same as project root. Exit.')

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
    if torch.cuda.is_available():
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.test.ddp.which_gpu).replace(' ', '').replace('[', '').replace(
            ']', '')
        mp.spawn(test, args=(config,), nprocs=config.test.ddp.nproc_this_node, join=True)
    else:
        raise ValueError('Please use GPU for testing!')


def test(local_rank, config):
    print(f'sys.argv[0]:{sys.argv[0]}')

    rank = config.test.ddp.rank_starts_from + local_rank

    hostname = socket.gethostname()
    if 'iesservergpu' in hostname:
        save_dir = f'/data/users/fu/APES/test_results/{config.wandb.name}/'
    else:
        save_dir = f'/home/team1/cwu/FuHaoWorkspace/test_results/{config.wandb.name}/'
    if rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # set files path
    if config.test.suffix.enable:
        artifacts_path = f'./artifacts/{config.wandb.name}_{config.test.suffix.remark}'
        backup_path = f'/data/users/wan/artifacts/{config.wandb.name}_{config.test.suffix.remark}'
    else:
        artifacts_path = f'./artifacts/{config.wandb.name}'
        backup_path = f'/data/users/wan/artifacts/{config.wandb.name}'
    zip_file_path = f'{artifacts_path}.zip'

    # process initialization
    os.environ['MASTER_ADDR'] = str(config.test.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(config.test.ddp.master_port)
    os.environ['WORLD_SIZE'] = str(config.test.ddp.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # gpu setting
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)  # which gpu is used by current process
    print(
        f'[init] pid: {os.getpid()} - global rank: {rank} - local rank: {local_rank} - cuda: {config.test.ddp.which_gpu[local_rank]}')

    # get datasets
    if config.datasets.dataset_name == 'shapenet_Yi650M':
        _, _, _, test_set = dataloader.get_shapenet_dataset_Yi650M(config.datasets.saved_path, config.datasets.mapping,
                                                                   config.train.dataloader.selected_points,
                                                                   config.train.dataloader.fps,
                                                                   config.train.dataloader.data_augmentation.enable,
                                                                   config.train.dataloader.data_augmentation.num_aug,
                                                                   config.train.dataloader.data_augmentation.jitter.enable,
                                                                   config.train.dataloader.data_augmentation.jitter.std,
                                                                   config.train.dataloader.data_augmentation.jitter.clip,
                                                                   config.train.dataloader.data_augmentation.rotate.enable,
                                                                   config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                   config.train.dataloader.data_augmentation.rotate.angle_range,
                                                                   config.train.dataloader.data_augmentation.translate.enable,
                                                                   config.train.dataloader.data_augmentation.translate.x_range,
                                                                   config.train.dataloader.data_augmentation.translate.y_range,
                                                                   config.train.dataloader.data_augmentation.translate.z_range,
                                                                   config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                   config.train.dataloader.data_augmentation.anisotropic_scale.x_range,
                                                                   config.train.dataloader.data_augmentation.anisotropic_scale.y_range,
                                                                   config.train.dataloader.data_augmentation.anisotropic_scale.z_range,
                                                                   config.train.dataloader.data_augmentation.anisotropic_scale.isotropic,
                                                                   config.test.dataloader.vote.enable,
                                                                   config.test.dataloader.vote.num_vote)
    elif config.datasets.dataset_name == 'shapenet_AnTao350M':
        _, _, _, test_set = dataloader.get_shapenet_dataset_AnTao350M(config.datasets.saved_path,
                                                                      config.train.dataloader.selected_points,
                                                                      config.train.dataloader.fps,
                                                                      config.train.dataloader.data_augmentation.enable,
                                                                      config.train.dataloader.data_augmentation.num_aug,
                                                                      config.train.dataloader.data_augmentation.jitter.enable,
                                                                      config.train.dataloader.data_augmentation.jitter.std,
                                                                      config.train.dataloader.data_augmentation.jitter.clip,
                                                                      config.train.dataloader.data_augmentation.rotate.enable,
                                                                      config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                      config.train.dataloader.data_augmentation.rotate.angle_range,
                                                                      config.train.dataloader.data_augmentation.translate.enable,
                                                                      config.train.dataloader.data_augmentation.translate.x_range,
                                                                      config.train.dataloader.data_augmentation.translate.y_range,
                                                                      config.train.dataloader.data_augmentation.translate.z_range,
                                                                      config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                      config.train.dataloader.data_augmentation.anisotropic_scale.x_range,
                                                                      config.train.dataloader.data_augmentation.anisotropic_scale.y_range,
                                                                      config.train.dataloader.data_augmentation.anisotropic_scale.z_range,
                                                                      config.train.dataloader.data_augmentation.anisotropic_scale.isotropic,
                                                                      config.test.dataloader.vote.enable,
                                                                      config.test.dataloader.vote.num_vote)
    elif config.datasets.dataset_name == 'shapenet_Normal':
        _, _, _, test_set = dataloader.get_shapenet_dataset_Normal(config.datasets, config.train.dataloader,
                                                                   config.test.dataloader.vote)
    else:
        raise ValueError('Not implemented!')

    # get sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    test_loader = torch.utils.data.DataLoader(test_set, config.test.dataloader.batch_size_per_gpu,
                                              num_workers=config.test.dataloader.num_workers, drop_last=True,
                                              prefetch_factor=config.test.dataloader.prefetch,
                                              pin_memory=config.test.dataloader.pin_memory, sampler=test_sampler)

    # get model
    # my_model = seg_model_inference_time.ShapeNetModel(config)
    # my_model = seg_model_inference_time.ShapeNetModel_to_2nd_ds(config)
    # my_model = seg_model_inference_time.ShapeNetModel_to_1st_ds(config)
    my_model = seg_model_inference_time.ShapeNetModel_ds_only(config)

    my_model.eval()
    my_model = my_model.to(device)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model)
    map_location = {'cuda:0': f'cuda:{local_rank}'}

    if config.feature_learning_block.samble_downsample.bin.dynamic_boundaries:
        state_dict = torch.load(f'{artifacts_path}/checkpoint.pt', map_location=map_location)
        my_model.load_state_dict(state_dict['model_state_dict'])

        config.feature_learning_block.samble_downsample.bin.dynamic_boundaries = False
        config.feature_learning_block.samble_downsample.bin.bin_boundaries = [
            bin_boundaries[0][0, 0, 0, 1:].tolist()
            for bin_boundaries in state_dict['bin_boundaries']]
    else:
        my_model.load_state_dict(torch.load(f'{artifacts_path}/checkpoint.pt', map_location=map_location))

    with torch.no_grad():

        # pbar = tqdm(total=len(test_loader))
        print('Testing...')
        #
        # time_begin = time.time()
        # for i, (samples, seg_labels, cls_label) in enumerate(test_loader):
        #     seg_labels, cls_label = seg_labels.to(device), cls_label.to(device)
        #     if config.test.dataloader.vote.enable:
        #         for samples_vote in samples:
        #             samples_vote = samples_vote.to(device)
        #             preds = my_model(samples_vote, cls_label)
        #     else:
        #         raise NotImplementedError
        # #     pbar.update(1)
        # # pbar.close()
        # time_end = time.time()
        # print(f'Interference time: {time_end - time_begin}s')

        interference_time = 0
        for i, (samples, seg_labels, cls_label) in enumerate(test_loader):
            seg_labels, cls_label = seg_labels.to(device), cls_label.to(device)
            if config.test.dataloader.vote.enable:
                for samples_vote in samples:
                    samples_vote = samples_vote.to(device)
                    preds, interference_time_one_batch = my_model(samples_vote, cls_label)
                    interference_time += interference_time_one_batch
            else:
                raise NotImplementedError
        print(f'Interference time: {interference_time}s')


if __name__ == '__main__':
    num_arguments = len(sys.argv)

    subprocess.run('nvidia-smi', shell=True, text=True, stdout=None, stderr=subprocess.PIPE)
    config = OmegaConf.load('configs/default.yaml')
    cmd_config = {
        'usr_config': 'configs/seg_boltzmannT01_bin6.yaml',
        'datasets': 'shapenet_AnTao350M',
        'wandb': {'name': '2024_05_02_02_52_Shapenet_Token_Std_boltzmann_T01_bin6_2'},
        'test': {'ddp': {'which_gpu': [3]}}
    }
    config = OmegaConf.merge(config, OmegaConf.create(cmd_config))

    dataset_config = OmegaConf.load(f'configs/datasets/{config.datasets}.yaml')
    dataset_config = OmegaConf.create({'datasets': dataset_config})
    config = OmegaConf.merge(config, dataset_config)

    main_without_Decorators(config)
    main_without_Decorators(config)
    main_without_Decorators(config)
    main_without_Decorators(config)
    main_without_Decorators(config)
    main_without_Decorators(config)
    main_without_Decorators(config)
    main_without_Decorators(config)
    main_without_Decorators(config)
    main_without_Decorators(config)
    main_without_Decorators(config)
