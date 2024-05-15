import wandb
from omegaconf import OmegaConf
from pathlib import Path
from utils import dataloader
from models import cls_model_inference_time
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import hydra
import subprocess
import datetime
import socket
import time

from utils.ops import reshape_gathered_variable, gather_variable_from_gpus
from utils.visualization import *
from utils.visualization_data_processing import *
from utils.check_config import set_config_run


@hydra.main(version_base=None, config_path="./configs", config_name="default.yaml")
def main_with_Decorators(config):
    main_without_Decorators(config)


def main_without_Decorators(config):
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

    if config.datasets.dataset_name == 'modelnet_AnTao420M':
        dataloader.download_modelnet_AnTao420M(config.datasets.url, config.datasets.saved_path)
    elif config.datasets.dataset_name == 'modelnet_Alignment1024':
        dataloader.download_modelnet_Alignment1024(config.datasets.url, config.datasets.saved_path)
    else:
        raise ValueError('Not implemented!')

    time_label = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    # multiprocessing for ddp

    if torch.cuda.is_available():
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.test.ddp.which_gpu).replace(' ', '').replace('[', '').replace(
            ']', '')
        mp.spawn(test, args=(config,), nprocs=config.test.ddp.nproc_this_node, join=True)
    else:
        raise ValueError('Please use GPU for testing!')


def test(local_rank, config):
    rank = config.test.ddp.rank_starts_from + local_rank

    hostname = socket.gethostname()
    if 'iesservergpu' in hostname:
        save_dir = f'/home/ies/fu/train_output/test_results/{config.wandb.name}/'  # f'/data/users/fu/APES/test_results/{config.wandb.name}/'
    elif hostname == 'LAPTOP-MPPIHOVR':
        save_dir = f'E:/datasets/APES/test_results/boltmannT/{config.wandb.name}/'
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
    if config.datasets.dataset_name == 'modelnet_AnTao420M':
        # _, test_set = dataloader.get_modelnet_dataset_AnTao420M(config.datasets.saved_path,
        _, test_set = dataloader.get_modelnet_dataset_AnTao420M(config.datasets.saved_path,  # TODO
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
                                                                config.train.dataloader.data_augmentation.anisotropic_scale.isotropic)
    elif config.datasets.dataset_name == 'modelnet_Alignment1024':
        _, test_set = dataloader.get_modelnet_dataset_Alignment1024(config.datasets.saved_path,
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
                                                                    config.train.dataloader.data_augmentation.anisotropic_scale.isotropic)
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
    # my_model = cls_model_inference_time.ModelNetModel(config)
    # my_model = cls_model_inference_time.ModelNetModel_input_to_2nd_downsample(config)
    # my_model = cls_model_inference_time.ModelNetModel_input_to_1st_downsample(config)
    # my_model = cls_model_inference_time.ModelNetModel_downsample_only(config)
    my_model = cls_model_inference_time.ModelNetModel_all_inference_time(config)

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

    # get loss function
    if config.test.label_smoothing:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=config.test.epsilon)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    # start test
    loss_list = []
    pred_list = []
    cls_label_list = []
    sample_list = []

    # sampling_score_list = []
    # idx_down_list = []
    # idx_in_bins_list = []
    # probability_of_bins_list = []

    # vis_test_gather_dict = vis_data_structure_init(config, based_config=True)
    # for idx_mode in vis_test_gather_dict["trained"].keys():
    #     vis_test_gather_dict["trained"][idx_mode] = [[] for _ in range(len(config.feature_learning_block.downsample.M))]
    # if config.test.visualize_downsampled_points.enable:
    #     for idx_mode in vis_test_gather_dict["ds_points"].keys():
    #         vis_test_gather_dict["ds_points"][idx_mode] = [[] for _ in
    #                                                        range(len(config.feature_learning_block.downsample.M))]
    # if config.test.visualize_attention_heatmap.enable:
    #     for idx_mode in vis_test_gather_dict["heatmap"].keys():
    #         vis_test_gather_dict["heatmap"][idx_mode] = [[] for _ in
    #                                                      range(len(config.feature_learning_block.downsample.M))]

    with torch.no_grad():

        print('Testing...')
        # time_begin = time.time()
        # for i, (samples, cls_labels) in enumerate(test_loader):
        #     samples, cls_labels = samples.to(device), cls_labels.to(device)
        #     preds = my_model(samples)
        #
        # time_end = time.time()
        # print(f'Interference time: {time_end - time_begin}s')

        # downsampling_time = 0
        # for i, (samples, cls_labels) in enumerate(test_loader):
        #     samples, cls_labels = samples.to(device), cls_labels.to(device)
        #     preds, downsampling_time_oneiter = my_model(samples)
        #     downsampling_time += downsampling_time_oneiter
        # print(f'Interference time: {downsampling_time}s')

        (model_inference_time_total, model_inference_time_start_to_1stds, model_inference_time_start_to_2ndds,
         model_inference_time_dsonly, batch_counter) = (0, 0, 0, 0, 0)
        for i, (samples, cls_labels) in enumerate(test_loader):
            samples, cls_labels = samples.to(device), cls_labels.to(device)
            preds, (
                model_inference_time_total_one_iter,
                model_inference_time_start_to_1stds_one_iter,
                model_inference_time_start_to_2ndds_one_iter,
                model_inference_time_dsonly_one_iter) = my_model(samples)

            model_inference_time_total += model_inference_time_total_one_iter
            model_inference_time_start_to_1stds += model_inference_time_start_to_1stds_one_iter
            model_inference_time_start_to_2ndds += model_inference_time_start_to_2ndds_one_iter
            model_inference_time_dsonly += model_inference_time_dsonly_one_iter
            batch_counter += 1

        model_inference_time_total /= batch_counter
        model_inference_time_start_to_1stds /= batch_counter
        model_inference_time_start_to_2ndds /= batch_counter
        model_inference_time_dsonly /= batch_counter

        print(
            f'{model_inference_time_total}\t'
            f'{model_inference_time_start_to_1stds}\t'
            f'{model_inference_time_start_to_2ndds}\t'
            f'{model_inference_time_dsonly}\n')


if __name__ == '__main__':
    num_arguments = len(sys.argv)

    if num_arguments > 1:
        main_with_Decorators()
    else:
        subprocess.run('nvidia-smi', shell=True, text=True, stdout=None, stderr=subprocess.PIPE)
        config = OmegaConf.load('configs/default.yaml')
        cmd_config = {
            'usr_config': 'configs/cls_boltzmannT01_bin6.yaml',
            'datasets': 'modelnet_AnTao420M',
            'wandb': {'name': '2024_04_28_02_15_Modelnet_Token_Std_boltzmann_T01_bin6_1'},
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
