from utils import dataloader, visualization
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
from utils import metrics, debug
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda import amp
import numpy as np
from pytorch3d.ops import sample_farthest_points as fps



@hydra.main(version_base=None, config_path="./configs", config_name="default.yaml")
def main(config):

    # check working directory
    try:
        assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    except:
        exit(f'Working directory is not the same as project root. Exit.')

    # overwrite the default config with user config
    if config.usr_config:
        usr_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, usr_config)

    if config.datasets.dataset_name == 'modelnet_AnTao420M':
        dataloader.download_modelnet_AnTao420M(config.datasets.url, config.datasets.saved_path)
    elif config.datasets.dataset_name == 'modelnet_Alignment1024':
        dataloader.download_modelnet_Alignment1024(config.datasets.url, config.datasets.saved_path)
    else:
        raise ValueError('Not implemented!')
    
    mp.set_start_method('spawn')
    
    rank = config.train.ddp.rank_starts_from + 0
    # process initialization
    os.environ['MASTER_ADDR'] = str(config.test.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(config.test.ddp.master_port)
    os.environ['WORLD_SIZE'] = str(config.test.ddp.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    
    # gpu setting
    device = f'cuda:0'
    torch.cuda.set_device(device)  # which gpu is used by current process
    
    _, test_set = dataloader.get_modelnet_dataset_AnTao420M(config.datasets.saved_path, config.train.dataloader.selected_points, config.train.dataloader.fps, config.train.dataloader.data_augmentation.enable, config.train.dataloader.data_augmentation.num_aug,
                                                                    config.train.dataloader.data_augmentation.jitter.enable, config.train.dataloader.data_augmentation.jitter.std, config.train.dataloader.data_augmentation.jitter.clip, config.train.dataloader.data_augmentation.rotate.enable, config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                    config.train.dataloader.data_augmentation.rotate.angle_range, config.train.dataloader.data_augmentation.translate.enable, config.train.dataloader.data_augmentation.translate.x_range,
                                                                    config.train.dataloader.data_augmentation.translate.y_range, config.train.dataloader.data_augmentation.translate.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                    config.train.dataloader.data_augmentation.anisotropic_scale.x_range, config.train.dataloader.data_augmentation.anisotropic_scale.y_range, config.train.dataloader.data_augmentation.anisotropic_scale.z_range)
    # get sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    test_loader = torch.utils.data.DataLoader(test_set, config.test.dataloader.batch_size_per_gpu, num_workers=config.test.dataloader.num_workers, drop_last=True, prefetch_factor=config.test.dataloader.prefetch, pin_memory=config.test.dataloader.pin_memory, sampler=test_sampler)


    
    # start test
    cls_label_list = []
    sample_list = []
    
    ds_type = "rs" # fps or rs

    ds_1_points_idx_gather_list = []
    ds_2_points_idx_gather_list = []
    ds_3_points_idx_gather_list = []
    ds_points_idx = []
    for i, (samples, cls_labels) in enumerate(test_loader):
        samples, cls_labels = samples.to(device), cls_labels.to(device) # 2 3 1024
        if ds_type == "fps": 
            for i in range(config.test.dataloader.batch_size_per_gpu):
                ds_1_points, index_1 = fps(samples.permute(0,2,1), K=1024, random_start_point=True)
                ds_2_points, index_2 = fps(ds_1_points, K=512, random_start_point=True)
                ds_3_points, index_3 = fps(ds_2_points, K=256, random_start_point=True)
                
                index_1 = index_1.unsqueeze(0)
                ds_1_points_idx_gather_list.append(index_1.detach().cpu())
                index_2 = index_2.unsqueeze(0)
                ds_2_points_idx_gather_list.append(index_2.detach().cpu())
                index_3 = index_3.unsqueeze(0)
                ds_3_points_idx_gather_list.append(index_3.detach().cpu())
        else:
            for _ in range(config.test.dataloader.batch_size_per_gpu):
                index_1 = torch.randperm(2048)[:1024].sort()[0]
                ds_1_points = samples[..., index_1]                
                index_temp = torch.randperm(1024)[:512].sort()[0]
                index_2 = index_1[index_temp]
                ds_2_points = samples[..., index_2]
                index_temp = torch.randperm(512)[:256].sort()[0]
                index_3 = index_2[index_temp]
                ds_3_points = samples[..., index_3]
                
                index_1 = index_1.unsqueeze(0).unsqueeze(0)
                ds_1_points_idx_gather_list.append(index_1)
                index_2 = index_2.unsqueeze(0).unsqueeze(0)
                ds_2_points_idx_gather_list.append(index_2)
                index_3 = index_3.unsqueeze(0).unsqueeze(0)
                ds_3_points_idx_gather_list.append(index_3)
                
        ds_1_points_idx = torch.concat(ds_1_points_idx_gather_list, dim=0).numpy()
        ds_2_points_idx = torch.concat(ds_2_points_idx_gather_list, dim=0).numpy()
        ds_3_points_idx = torch.concat(ds_3_points_idx_gather_list, dim=0).numpy()
                
        cls_label_gather_list = [torch.empty_like(cls_labels).to(device) for _ in range(config.test.ddp.nproc_this_node)]
        sample_gather_list = [torch.empty_like(samples).to(device) for _ in range(config.test.ddp.nproc_this_node)]
        
        torch.distributed.all_gather(cls_label_gather_list, cls_labels)
        torch.distributed.all_gather(sample_gather_list, samples)
        
        # ds_points = torch.concat((ds_1_points,ds_2_points), dim=0)
        cls_labels = torch.concat(cls_label_gather_list, dim=0)
        cls_label_list.append(torch.max(cls_labels, dim=1)[1].detach().cpu().numpy())
        samples = torch.concat(sample_gather_list, dim=0)
        sample_list.append(samples.permute(0, 2, 1).detach().cpu().numpy())
        # ds_points = torch.concat((ds_1_points,ds_2_points), dim=0)
    ds_points_idx.append(np.array(ds_1_points_idx))
    ds_points_idx.append(np.array(ds_2_points_idx))
    ds_points_idx.append(np.array(ds_3_points_idx))
    cls_labels = np.concatenate(cls_label_list, axis=0)
    samples = np.concatenate(sample_list, axis=0)
    
    visualization.visualize_modelnet_downsampled_points_rs_fps(config, samples, ds_points_idx, cls_labels, ds_type)
    print("finished")

if __name__ == '__main__':
    main()
