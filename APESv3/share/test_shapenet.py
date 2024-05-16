import wandb
import hydra
import torch
import os
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import pkbar
from omegaconf import OmegaConf
from pathlib import Path
from utils import dataloader, metrics
from models import seg_model
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
    if config.datasets.dataset_name == 'shapenet_Yi650M':
        _, _, _, test_set = dataloader.get_shapenet_dataset_Yi650M(config.datasets.saved_path, config.datasets.mapping, config.train.dataloader.selected_points, config.train.dataloader.fps, config.train.dataloader.data_augmentation.enable, config.train.dataloader.data_augmentation.num_aug, config.train.dataloader.data_augmentation.jitter.enable,
                                                                   config.train.dataloader.data_augmentation.jitter.std, config.train.dataloader.data_augmentation.jitter.clip, config.train.dataloader.data_augmentation.rotate.enable, config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                   config.train.dataloader.data_augmentation.rotate.angle_range, config.train.dataloader.data_augmentation.translate.enable, config.train.dataloader.data_augmentation.translate.x_range,
                                                                   config.train.dataloader.data_augmentation.translate.y_range, config.train.dataloader.data_augmentation.translate.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                   config.train.dataloader.data_augmentation.anisotropic_scale.x_range, config.train.dataloader.data_augmentation.anisotropic_scale.y_range, config.train.dataloader.data_augmentation.anisotropic_scale.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.isotropic,
                                                                   config.test.dataloader.vote.enable, config.test.dataloader.vote.num_vote)
    elif config.datasets.dataset_name == 'shapenet_AnTao350M':
        _, _, _, test_set = dataloader.get_shapenet_dataset_AnTao350M(config.datasets.saved_path, config.train.dataloader.selected_points, config.train.dataloader.fps, config.train.dataloader.data_augmentation.enable, config.train.dataloader.data_augmentation.num_aug, config.train.dataloader.data_augmentation.jitter.enable,
                                                                      config.train.dataloader.data_augmentation.jitter.std, config.train.dataloader.data_augmentation.jitter.clip, config.train.dataloader.data_augmentation.rotate.enable, config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                      config.train.dataloader.data_augmentation.rotate.angle_range, config.train.dataloader.data_augmentation.translate.enable, config.train.dataloader.data_augmentation.translate.x_range,
                                                                      config.train.dataloader.data_augmentation.translate.y_range, config.train.dataloader.data_augmentation.translate.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                      config.train.dataloader.data_augmentation.anisotropic_scale.x_range, config.train.dataloader.data_augmentation.anisotropic_scale.y_range, config.train.dataloader.data_augmentation.anisotropic_scale.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.isotropic,
                                                                      config.test.dataloader.vote.enable, config.test.dataloader.vote.num_vote)
    elif config.datasets.dataset_name == 'shapenet_Normal':
        _, _, _, test_set = dataloader.get_shapenet_dataset_Normal(config.datasets, config.train.dataloader, config.test.dataloader.vote)
    else:
        raise ValueError('Not implemented!')

    # get sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    test_loader = torch.utils.data.DataLoader(test_set, config.test.dataloader.batch_size_per_gpu, num_workers=config.test.dataloader.num_workers, drop_last=True, prefetch_factor=config.test.dataloader.prefetch, pin_memory=config.test.dataloader.pin_memory, sampler=test_sampler)

    # get model
    my_model = seg_model.ShapeNetModel(config)
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
    seg_label_list = []
    cls_label_list = []
    sample_list = []
    pred_novo_list = []
    loss_novo_list = []
    vis_test_gather_dict = vis_data_structure_init(config, based_config=True)
    for mode in vis_test_gather_dict["trained"].keys():
        vis_test_gather_dict["trained"][mode] = [[] for _ in range(len(config.neighbor2point_block.downsample.M))]
    
    with torch.no_grad():
        if rank == 0:
            print(f'Print Results: {config.test.print_results} - Visualize Predictions: {config.test.visualize_preds.enable} - Visualize Downsampled Points: {config.test.visualize_downsampled_points.enable}')
            pbar = pkbar.Pbar(name='Start testing, please wait...', target=len(test_loader))
        for i, (samples, seg_labels, cls_label) in enumerate(test_loader):
            seg_labels, cls_label = seg_labels.to(device), cls_label.to(device)
            if config.test.dataloader.vote.enable:
                preds_list = []
                for samples_vote in samples:
                    samples_vote = samples_vote.to(device)
                    preds = my_model(samples_vote, cls_label)
                    preds_list.append(preds)
                preds = torch.mean(torch.stack(preds_list), dim=0)
                samples = samples[0].to(device)
                
                # collect no voting result
                preds_novo = preds_list[0]
                loss_novo = loss_fn(preds_novo, seg_labels)
                pred_novo_gather_list = [torch.empty_like(preds_novo).to(device) for _ in range(config.test.ddp.nproc_this_node)]
                torch.distributed.all_gather(pred_novo_gather_list, preds_novo)
                if rank == 0:
                    preds_novo = torch.concat(pred_novo_gather_list, dim=0)
                    pred_novo_list.append(torch.max(preds_novo.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                    loss_novo /= config.test.ddp.nproc_this_node
                    loss_novo_list.append(loss_novo.detach().cpu().numpy())
            else:
                samples = samples.to(device)    
                preds = my_model(samples, cls_label)
            loss = loss_fn(preds, seg_labels)

            # collect the result among all gpus
            pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            seg_label_gather_list = [torch.empty_like(seg_labels).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            cls_label_gather_list = [torch.empty_like(cls_label).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            sample_gather_list = [torch.empty_like(samples).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            
            vis_test_gather_dict = vis_data_gather(config, my_model, device, rank, vis_test_gather_dict)
            
            torch.distributed.all_gather(pred_gather_list, preds)
            torch.distributed.all_gather(seg_label_gather_list, seg_labels)
            torch.distributed.all_gather(cls_label_gather_list, cls_label)
            torch.distributed.all_gather(sample_gather_list, samples)
            torch.distributed.all_reduce(loss)
            if rank == 0:
                preds = torch.concat(pred_gather_list, dim=0)
                pred_list.append(torch.max(preds.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                seg_labels = torch.concat(seg_label_gather_list, dim=0)
                seg_label_list.append(torch.max(seg_labels.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                cls_label = torch.concat(cls_label_gather_list, dim=0)
                cls_label_list.append(torch.max(cls_label[:, :, 0], dim=1)[1].detach().cpu().numpy())
                samples = torch.concat(sample_gather_list, dim=0) # samples.shape == (B, C, N)
                sample_list.append(samples[:, :3, :].permute(0, 2, 1).detach().cpu().numpy())
                loss /= config.test.ddp.nproc_this_node
                loss_list.append(loss.detach().cpu().numpy())
                pbar.update(i)

    if rank == 0:
        preds = np.concatenate(pred_list, axis=0)
        seg_labels = np.concatenate(seg_label_list, axis=0)
        cls_label = np.concatenate(cls_label_list, axis=0)
        samples = np.concatenate(sample_list, axis=0)
        if config.test.dataloader.vote.enable:
            preds_novo = np.concatenate(pred_novo_list, axis=0)
            shape_ious_novo = metrics.calculate_shape_IoU(preds_novo, seg_labels, cls_label, config.datasets.mapping)
            category_iou_novo = metrics.calculate_category_IoU(shape_ious_novo, cls_label, config.datasets.mapping)
            miou_novo = sum(shape_ious_novo) / len(shape_ious_novo)
            category_miou_novo = sum(list(category_iou_novo.values())) / len(list(category_iou_novo.values()))
            loss_novo = sum(loss_novo_list) / len(loss_novo_list)
            with open(f'{artifacts_path}/results.txt', 'a') as f:
                f.write(f'loss_novo: {loss_novo}\n')
                f.write(f'mIoU_novo: {miou_novo}\n')
                f.write(f'category_mIoU_novo: {category_miou_novo}\n\n')
                for category in list(category_iou_novo.keys()):
                    f.write(f'{category}_novo: {category_iou_novo[category]}\n')
                f.write('\n\n\n\n')
            if config.test.print_results:
                print(f'loss_novo: {loss_novo}')
                print(f'mIoU_novo: {miou_novo}')
                print(f'category_mIoU_novo: {category_miou_novo}')
                for category in list(category_iou_novo.keys()):
                    print(f'{category}_novo: {category_iou_novo[category]}')
        
        vis_concat_dict = vis_data_structure_init(config, based_config=True)
        vis_concat_dict = vis_data_concat(len(config.neighbor2point_block.downsample.M), vis_concat_dict, vis_test_gather_dict)
        
        # calculate metrics
        shape_ious = metrics.calculate_shape_IoU(preds, seg_labels, cls_label, config.datasets.mapping)
        category_iou = metrics.calculate_category_IoU(shape_ious, cls_label, config.datasets.mapping)
        miou = sum(shape_ious) / len(shape_ious)
        category_miou = sum(list(category_iou.values())) / len(list(category_iou.values()))
        loss = sum(loss_list) / len(loss_list)
        with open(f'{artifacts_path}/results.txt', 'a') as f:
            f.write(f'loss: {loss}\n')
            f.write(f'mIoU: {miou}\n')
            f.write(f'category_mIoU: {category_miou}\n\n')
            for category in list(category_iou.keys()):
                f.write(f'{category}: {category_iou[category]}\n')
        if config.test.print_results:
            print(f'loss: {loss}')
            print(f'mIoU: {miou}')
            print(f'category_mIoU: {category_miou}')
            for category in list(category_iou.keys()):
                print(f'{category}: {category_iou[category]}')

        # generating visualized downsampled points files
        if config.test.visualize_downsampled_points.enable:
            if config.neighbor2point_block.downsample.bin.enable[0]:
                visualize_shapenet_downsampled_points_bin(config, samples, vis_concat_dict["trained"]["idx"], vis_concat_dict["trained"]["bin_prob"], cls_label, shape_ious, artifacts_path)
            else:
                visualize_shapenet_downsampled_points(config, samples, vis_concat_dict["trained"]["idx"], cls_label, shape_ious, artifacts_path)
        # generating visualized prediction files
        if config.test.visualize_preds.enable:
            visualize_shapenet_predictions(config, samples, preds, seg_labels, cls_label, shape_ious, vis_concat_dict["trained"]["idx"], artifacts_path)

        # save_backup(artifacts_path, zip_file_path, backup_path)

if __name__ == '__main__':
    main()
