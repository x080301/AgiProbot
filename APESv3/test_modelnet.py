import wandb
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
import os
import numpy as np
from utils import dataloader, metrics
from models import cls_model
import torch.multiprocessing as mp
import torch.distributed as dist
import pkbar
from utils.visualization import *
from utils.visualization_data_processing import *
from utils.check_config import set_config_run
from utils.save_backup import save_backup


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
    if torch.cuda.is_available():
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.test.ddp.which_gpu).replace(' ', '').replace('[', '').replace(']', '')
        mp.spawn(test, args=(config,), nprocs=config.test.ddp.nproc_this_node, join=True)
    else:
        raise ValueError('Please use GPU for testing!')


def test(local_rank, config):

    rank = config.test.ddp.rank_starts_from + local_rank
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
    print(f'[init] pid: {os.getpid()} - global rank: {rank} - local rank: {local_rank} - cuda: {config.test.ddp.which_gpu[local_rank]}')

    # get datasets
    if config.datasets.dataset_name == 'modelnet_AnTao420M':
         _, test_set = dataloader.get_modelnet_dataset_AnTao420M(config.datasets.saved_path, config.train.dataloader.selected_points, config.train.dataloader.fps, config.train.dataloader.data_augmentation.enable, config.train.dataloader.data_augmentation.num_aug, config.train.dataloader.data_augmentation.jitter.enable,
                                                                 config.train.dataloader.data_augmentation.jitter.std, config.train.dataloader.data_augmentation.jitter.clip, config.train.dataloader.data_augmentation.rotate.enable, config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                 config.train.dataloader.data_augmentation.rotate.angle_range, config.train.dataloader.data_augmentation.translate.enable, config.train.dataloader.data_augmentation.translate.x_range,
                                                                 config.train.dataloader.data_augmentation.translate.y_range, config.train.dataloader.data_augmentation.translate.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                 config.train.dataloader.data_augmentation.anisotropic_scale.x_range, config.train.dataloader.data_augmentation.anisotropic_scale.y_range, config.train.dataloader.data_augmentation.anisotropic_scale.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.isotropic)
    elif config.datasets.dataset_name == 'modelnet_Alignment1024':
        _, test_set = dataloader.get_modelnet_dataset_Alignment1024(config.datasets.saved_path, config.train.dataloader.selected_points, config.train.dataloader.fps, config.train.dataloader.data_augmentation.enable, config.train.dataloader.data_augmentation.num_aug,
                                                                    config.train.dataloader.data_augmentation.jitter.enable, config.train.dataloader.data_augmentation.jitter.std, config.train.dataloader.data_augmentation.jitter.clip, config.train.dataloader.data_augmentation.rotate.enable, config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                    config.train.dataloader.data_augmentation.rotate.angle_range, config.train.dataloader.data_augmentation.translate.enable, config.train.dataloader.data_augmentation.translate.x_range,
                                                                    config.train.dataloader.data_augmentation.translate.y_range, config.train.dataloader.data_augmentation.translate.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                    config.train.dataloader.data_augmentation.anisotropic_scale.x_range, config.train.dataloader.data_augmentation.anisotropic_scale.y_range, config.train.dataloader.data_augmentation.anisotropic_scale.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.isotropic)
    else:
        raise ValueError('Not implemented!')

    # get sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    test_loader = torch.utils.data.DataLoader(test_set, config.test.dataloader.batch_size_per_gpu, num_workers=config.test.dataloader.num_workers, drop_last=True, prefetch_factor=config.test.dataloader.prefetch, pin_memory=config.test.dataloader.pin_memory, sampler=test_sampler)

    # get model
    my_model = cls_model.ModelNetModel(config)
    my_model.eval()
    my_model = my_model.to(device)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model)
    map_location = {'cuda:0': f'cuda:{local_rank}'}
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
    
    vis_test_gather_dict = vis_data_structure_init(config, based_config=True)
    for idx_mode in vis_test_gather_dict["trained"].keys():
        vis_test_gather_dict["trained"][idx_mode] = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    if config.test.visualize_downsampled_points.enable:
        for idx_mode in vis_test_gather_dict["ds_points"].keys():
            vis_test_gather_dict["ds_points"][idx_mode] = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    if config.test.visualize_attention_heatmap.enable:
        for idx_mode in vis_test_gather_dict["heatmap"].keys():
            vis_test_gather_dict["heatmap"][idx_mode] = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]

    with torch.no_grad():
        if rank == 0:
            print(f'Print Results: {config.test.print_results} - Visualize Downsampled Points: {config.test.visualize_downsampled_points.enable} - Visualize Heatmap: {config.test.visualize_attention_heatmap.enable}')
            pbar = pkbar.Pbar(name='Start testing, please wait...', target=len(test_loader))
        for i, (samples, cls_labels) in enumerate(test_loader):
            samples, cls_labels = samples.to(device), cls_labels.to(device)
            preds = my_model(samples)
            
            if config.train.aux_loss.enable:
                preds = preds[-1]
                loss = loss_fn(preds, cls_labels)
            else:
                loss = loss_fn(preds, cls_labels)

            # collect the result among all gpus
            pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            cls_label_gather_list = [torch.empty_like(cls_labels).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            sample_gather_list = [torch.empty_like(samples).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            
            vis_test_gather_dict = vis_data_gather(config, my_model, device, rank, vis_test_gather_dict)
            torch.distributed.all_gather(pred_gather_list, preds)
            torch.distributed.all_gather(cls_label_gather_list, cls_labels)
            torch.distributed.all_gather(sample_gather_list, samples)
            torch.distributed.all_reduce(loss)
            
            if rank == 0:
                preds = torch.concat(pred_gather_list, dim=0)
                pred_list.append(torch.max(preds, dim=1)[1].detach().cpu().numpy())
                cls_labels = torch.concat(cls_label_gather_list, dim=0)
                cls_label_list.append(torch.max(cls_labels, dim=1)[1].detach().cpu().numpy())
                samples = torch.concat(sample_gather_list, dim=0)
                sample_list.append(samples.permute(0, 2, 1).detach().cpu().numpy())
                loss /= config.test.ddp.nproc_this_node
                loss_list.append(loss.detach().cpu().numpy())
                pbar.update(i)

    if rank == 0:
        preds = np.concatenate(pred_list, axis=0)
        cls_labels = np.concatenate(cls_label_list, axis=0)
        samples = np.concatenate(sample_list, axis=0)
        
        vis_concat_dict = vis_data_structure_init(config, based_config=True)
        vis_concat_dict = vis_data_concat(len(config.neighbor2point_block.downsample.M), vis_concat_dict, vis_test_gather_dict)

        # calculate metrics
        acc = metrics.calculate_accuracy(preds, cls_labels)
        category_acc = metrics.calculate_category_accuracy(preds, cls_labels, config.datasets.mapping)
        loss = sum(loss_list) / len(loss_list)
        if config.test.print_results:
            print(f'loss: {loss}')
            print(f'accuracy: {acc}')
            for category in list(category_acc.keys()):
                print(f'{category}: {category_acc[category]}')
        with open(f'{artifacts_path}/metrics.txt', 'w') as f:
            f.write(f'loss: {loss}\n')
            f.write(f'accuracy: {acc}\n')
            for category in list(category_acc.keys()):
                f.write(f'{category}: {category_acc[category]}\n')
            f.close()
        
        # generating visualized downsampled points files
        if config.test.visualize_downsampled_points.enable:
            ds_path = f'{artifacts_path}/vis_ds_points'
            if os.path.exists(ds_path):
                shutil.rmtree(ds_path)
            for idx_mode in vis_test_gather_dict["ds_points"].keys():
                if len(vis_test_gather_dict["ds_points"].keys()) == 1 and not config.test.few_points.enable:
                    index = vis_concat_dict["trained"]["idx"]
                else:
                    index = vis_concat_dict["ds_points"][idx_mode]
                if config.test.few_points.enable:
                    visualize_modelnet_downsampled_points_few_points(config, samples, index, cls_labels, idx_mode, artifacts_path)
                else:
                    if config.neighbor2point_block.downsample.bin.enable[0]:
                        visualize_modelnet_downsampled_points_bin(config, samples, index, vis_concat_dict["trained"]["bin_prob"], cls_labels, idx_mode, artifacts_path)
                    else:
                        visualize_modelnet_downsampled_points(config, samples, index, cls_labels, idx_mode, artifacts_path)
        # generating visualized heatmap files
        if config.test.visualize_attention_heatmap.enable:
            hm_path = f'{artifacts_path}/vis_heatmap'
            if os.path.exists(hm_path):
                shutil.rmtree(hm_path)
            for idx_mode in vis_test_gather_dict["heatmap"].keys():
                if len(vis_test_gather_dict["heatmap"].keys()) == 1:
                    attention_map = vis_concat_dict["trained"]["attention_point_score"]
                else:
                    attention_map = vis_concat_dict["heatmap"][idx_mode]
                visualize_modelnet_heatmap_mode(config, samples, attention_map, cls_labels, idx_mode, artifacts_path)
            if config.neighbor2point_block.downsample.boltzmann.enable[0]:
                aps_boltz = vis_concat_dict["trained"]["aps_boltz"]
                visualize_modelnet_heatmap_mode(config, samples, aps_boltz, cls_labels, 'trained_boltzmann', artifacts_path)
        if config.test.visualize_combine.enable:
            assert config.test.visualize_downsampled_points.enable or config.test.visualize_attention_heatmap.enable, "At least one of visualize_downsampled_points or visualize_attention_heatmap must be enabled."
            visualize_modelnet_combine(config, artifacts_path)
        
        # storage and backup
        # save_backup(artifacts_path, zip_file_path, backup_path)

if __name__ == '__main__':
    main()
