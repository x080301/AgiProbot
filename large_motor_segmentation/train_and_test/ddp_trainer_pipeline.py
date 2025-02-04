import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import platform
import datetime
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import copy
from contextlib import nullcontext

import models.attention
import utilities.loss_calculation
from utilities.config import get_parser
from data_preprocess.data_loader import MotorDataset
from utilities.lr_scheduler import CosineAnnealingWithWarmupLR
from utilities import util
from utilities.util import save_tensorboard_log

from models.pct import PCTPipeline
from models.pointnet import PointNetSegmentation
from models.pointnet2 import PointNet2Segmentation
from models.dgcnn import DGCNN_semseg

files_to_save = ['config', 'data_preprocess', 'ideas', 'models', 'train_and_test', 'utilities',
                 'train.py', 'train_line.py', 'best_m.pth']


def init_training(train_txt, config_dir='config/binary_segmentation.yaml', valid_motors=None, local_training=False,
                  direction=None):
    # ******************* #
    # load arguments
    # ******************* #
    args = get_parser(config_dir=config_dir)
    print("use", torch.cuda.device_count(), "GPUs for training")

    if args.random_seed == 0:
        random_seed = int(time.time())
    else:
        random_seed = args.train.random_seed

    args.train_batch_size = int(args.train_batch_size / args.accumulation_steps)

    # ******************* #
    # local or server?
    # ******************* #
    system_type = platform.system().lower()  # 'windows' or 'linux'
    is_local = True if system_type == 'windows' else False
    if is_local:
        if local_training:
            data_set_direction = args.local_training_data_dir
        elif args.finetune == 0:
            data_set_direction = args.pretrain_local_data_dir
        else:
            data_set_direction = args.fine_tune_local_data_dir
        # 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_npy'
    else:
        if args.finetune == 0:
            data_set_direction = args.pretrain_server_data_dir
        else:
            data_set_direction = args.fine_tune_server_data_dir

    # ******************* #
    # make directions
    # ******************* #

    if is_local:
        if direction is None:
            direction = 'outputs/'

        if valid_motors is None:
            direction += args.titel + '_' + datetime.datetime.now().strftime(
                '%Y_%m_%d_%H_%M')
        else:
            direction += args.titel + '_' + datetime.datetime.now().strftime(
                '%Y_%m_%d_%H_%M') + '_' + valid_motors.split('&')[0] + '-' + valid_motors.split('&')[1]
    else:
        if direction is None:
            direction = '/data/users/fu/'
        if valid_motors is None:
            direction += args.titel + '_outputs/' + \
                         datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        else:
            direction += args.titel + '_outputs/' + \
                         datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') + '_' + valid_motors.split('&')[0] + '-' + \
                         valid_motors.split('&')[1]

    if not os.path.exists(direction + '/checkpoints'):
        os.makedirs(direction + '/checkpoints')
    if not os.path.exists(direction + '/train_log'):
        os.makedirs(direction + '/train_log')
    if not os.path.exists(direction + '/tensorboard_log'):
        os.makedirs(direction + '/tensorboard_log')

    # ******************* #
    # save mode and parameters
    # ******************* #
    if train_txt is not None:
        print(train_txt)
        shutil.copyfile(train_txt, direction + '/train_log/' + train_txt.split('/')[-1])
    for file_name in files_to_save:
        if '.' in file_name:
            shutil.copyfile(file_name, direction + '/train_log/' + file_name.split('/')[-1])
        else:
            shutil.copytree(file_name, direction + '/train_log/' + file_name.split('/')[-1])

    with open(direction + '/train_log/' + 'random_seed_' + str(random_seed) + '.txt', 'w') as f:
        f.write('')

    save_direction = direction

    # ******************* #
    # load data set
    # ******************* #
    print("start loading training data ...")

    train_dataset = MotorDataset(mode='train',
                                 data_dir=data_set_direction,
                                 num_class=args.num_segmentation_type, num_points=args.npoints,
                                 # 4096
                                 test_area='Validation', sample_rate=args.sample_rate,
                                 valid_motors=valid_motors)
    print("start loading test data ...")
    valid_dataset = MotorDataset(mode='valid',
                                 data_dir=data_set_direction,
                                 num_class=args.num_segmentation_type, num_points=args.npoints,
                                 # 4096
                                 test_area='Validation', sample_rate=1.0, valid_motors=valid_motors)

    # ******************* #
    # dpp
    # ******************* #
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    return args, random_seed, is_local, save_direction, train_dataset, valid_dataset


def train_ddp(rank, world_size, args, random_seed, is_local, save_direction, train_dataset, valid_dataset, start_time,
              local_training):
    best_mIoU = 0
    checkpoints_direction = save_direction + '/checkpoints/'
    torch.manual_seed(random_seed)

    # ******************* #
    # dpp and load ML model
    # ******************* #
    torch.cuda.set_device(rank)
    backend = 'gloo' if is_local else 'nccl'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # ******************* #
    # load ML model
    # ******************* #
    if args.model_para.model == 'pct':
        model = PCTPipeline(args)
    elif args.model_para.model == 'pointnet':
        model = PointNetSegmentation(num_segmentation_class=args.num_segmentation_type, feature_transform=True)
    elif args.model_para.model == 'pointnet2':
        model = PointNet2Segmentation(num_segmentation_class=args.num_segmentation_type)
    elif args.model_para.model == 'dgcnn':
        model = DGCNN_semseg(num_classes=args.num_segmentation_type)
    else:
        raise NotImplemented

    model.to(rank)

    if args.finetune == 1:
        if rank == 0:
            if os.path.exists(args.pretrained_model_path):
                checkpoint = torch.load(args.pretrained_model_path,
                                        map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
                print('Use pretrained model for fine tune')
            else:
                print('no exiting pretrained model')
                exit(-1)

            start_epoch = checkpoint['epoch']

            if local_training:
                end_epoch = start_epoch + args.epochs
            elif is_local:
                end_epoch = start_epoch + 2
            else:
                end_epoch = start_epoch + args.epochs

            if 'mIoU' in checkpoint:
                print('train begin at %dth epoch with mIoU %.6f' % (start_epoch, checkpoint['mIoU']))
            else:
                print('train begin with %dth epoch' % start_epoch)
        else:
            if os.path.exists(args.pretrained_model_path):
                checkpoint = torch.load(args.pretrained_model_path,
                                        map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
            else:
                exit(-1)

            start_epoch = checkpoint['epoch']

            if local_training:
                end_epoch = start_epoch + args.epochs
            elif is_local:
                end_epoch = start_epoch + 2
            else:
                end_epoch = start_epoch + args.epochs

        state_dict = checkpoint['model_state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k[0:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)  # .to(rank)
    else:
        start_epoch = 0
        if local_training:
            end_epoch = start_epoch + args.epochs
        elif is_local:
            end_epoch = start_epoch + 2
        else:
            end_epoch = start_epoch + args.epochs

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        log_writer = SummaryWriter(save_direction + '/tensorboard_log')

    # ******************* #
    # load dataset
    # ******************* #
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank
                                                                    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank
                                                                    )

    para_workers = 0 if is_local else 8
    train_loader = DataLoader(train_dataset,
                              num_workers=para_workers,
                              batch_size=args.train_batch_size,
                              shuffle=False,
                              drop_last=True,
                              pin_memory=True,
                              sampler=train_sampler
                              )
    num_train_batch = len(train_loader)

    validation_loader = DataLoader(valid_dataset,
                                   num_workers=para_workers,
                                   pin_memory=True,
                                   sampler=valid_sampler,
                                   batch_size=args.test_batch_size,
                                   shuffle=False,
                                   drop_last=True
                                   )
    num_valid_batch = len(validation_loader)

    # ******************* #
    # opt
    # ******************* #
    if args.use_sgd:
        opt = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}],
                              lr=args.lr,
                              momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                               eta_min=args.end_lr)
    elif args.scheduler == 'step':
        if args.model_para.model == 'pointnet' or args.model_para.model == 'pointnet2':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, 0.7, -1)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, 0.1, args.epochs)
    elif args.scheduler == 'cos_warmupLR':
        scheduler = CosineAnnealingWithWarmupLR(opt,
                                                T_max=args.epochs - args.cos_warmupLR.warmup_epochs,
                                                eta_min=args.end_lr,
                                                warmup_init_lr=args.end_lr,
                                                warmup_epochs=args.cos_warmupLR.warmup_epochs)

    else:
        print('no scheduler called' + args.scheduler)
        exit(-1)

    # ******************* #
    # loss function and weights
    # ******************* #
    if args.model_para.model == 'pct':
        criterion = utilities.loss_calculation.loss_calculation
    elif args.model_para.model == 'pointnet':
        criterion = utilities.loss_calculation.loss_calculation_pointnet
    elif args.model_para.model == 'pointnet2':
        criterion = utilities.loss_calculation.loss_calculation_cross_entropy
    elif args.model_para.model == 'dgcnn':
        criterion = utilities.loss_calculation.loss_calculation_cross_entropy
    else:
        raise NotImplemented

    # print(weights)
    # percentage = torch.Tensor(train_dataset.persentage_each_type).cuda()
    # scale = weights * percentage
    # scale = 1 / scale.sum()
    # # print(scale)
    # weights *= scale
    # print(weights)
    weights = torch.Tensor(train_dataset.label_weights).to(rank)
    if args.use_class_weight == 0:
        for i in range(args.num_segmentation_type):
            weights[i] = 1
    elif args.use_class_weight == 1:
        pass
    elif args.use_class_weight == 0.5:
        weights = torch.pow(weights, 1 / 2)
    elif args.use_class_weight == 0.33:
        weights = torch.pow(weights, 1 / 3)
    else:
        raise NotImplemented

    if rank == 0:
        print('train %d epochs' % (end_epoch - start_epoch))
    for epoch in range(start_epoch, end_epoch):

        if rank == 0:
            running_time = (time.time() - start_time) / 60
            log_writer.add_scalar('running time', running_time, epoch)
            print('running time: %.2f ' % running_time)

        # ******************* #
        # train
        # ******************* #
        if rank == 0:
            print('\n-----train-----')
        train_sampler.set_epoch(epoch)
        model.train()

        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(args.num_segmentation_type)]
        total_correct_class = [0 for _ in range(args.num_segmentation_type)]
        total_iou_deno_class = [0 for _ in range(args.num_segmentation_type)]

        if rank == 0:
            tqdm_structure = tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9)
        else:
            tqdm_structure = enumerate(train_loader)
        for i, (points, target) in tqdm_structure:
            '''
            points: (B,N,3)
            target: (B,N)
            '''
            # ******************* #
            # forwards
            # ******************* #
            points, target = points.to(rank, non_blocking=True), target.to(rank, non_blocking=True)
            points = models.attention.normalize_data(points)

            # rotation augmentation
            points, _ = models.attention.rotate_per_batch(points, None)

            points = points.permute(0, 2, 1).float()
            batch_size = points.size()[0]

            mcontext = model.no_sync if world_size > 1 and (i + 1) % args.accumulation_steps != 0 else nullcontext
            #
            with mcontext():

                seg_pred, trans = model(points)  # (B,segment_type,N),(B,3,3)

                # print(seg_pred)

                # ******************* #
                # backwards
                # ******************* #
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()  # (B,segment_type,N) -> (B,N,segment_type)

                batch_label = target.view(-1, 1)[:, 0].data

                if args.model_para.model == 'pct':
                    loss = criterion(seg_pred.view(-1, args.num_segmentation_type), target.view(-1, 1).squeeze(),
                                     weights, using_weight=args.use_class_weight)  # a scalar
                    if not args.stn_loss_weight == 0:
                        loss = loss + util.feature_transform_reguliarzer(trans) * args.stn_loss_weight

                elif args.model_para.model == 'pointnet':
                    if args.use_class_weight == 0:
                        weights = None
                    loss = criterion(seg_pred.view(-1, args.num_segmentation_type),
                                     target.view(-1, 1).squeeze(),
                                     trans,
                                     weights)
                elif args.model_para.model == 'pointnet2':
                    if args.use_class_weight == 0:
                        weights = None
                    loss = criterion(seg_pred.view(-1, args.num_segmentation_type),
                                     target.view(-1, 1).squeeze(),
                                     weights)
                elif args.model_para.model == 'dgcnn':
                    if args.use_class_weight == 0:
                        weights = None
                    loss = criterion(seg_pred.view(-1, args.num_segmentation_type),
                                     target.view(-1, 1).squeeze(), weights)

                else:
                    raise NotImplemented

                loss = loss / args.accumulation_steps
                loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                opt.step()
                opt.zero_grad()

            # ******************* #
            # further calculation
            # ******************* #
            seg_pred = seg_pred.contiguous().view(-1, args.num_segmentation_type)
            # _                                                      (batch_size*num_points , num_class)
            pred_choice = seg_pred.data.max(1)[1]  # array(batch_size*num_points)
            correct = torch.sum(pred_choice == batch_label)
            # when a np arraies have same shape, a==b means in conrrespondending position it equals to one,when they are identical
            total_correct += correct
            total_seen += (batch_size * args.npoints)
            loss_sum += loss
            for l in range(args.num_segmentation_type):
                total_seen_class[l] += torch.sum((batch_label == l))
                total_correct_class[l] += torch.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class[l] += torch.sum(((pred_choice == l) | (batch_label == l)))

        IoUs = (torch.tensor(total_correct_class) / (torch.tensor(total_iou_deno_class).float() + 1e-6)).to(
            rank)
        mIoU = IoUs.sum() / args.num_existing_type  # torch.mean(IoUs)
        torch.distributed.all_reduce(IoUs)
        torch.distributed.all_reduce(mIoU)
        IoUs /= float(world_size)
        mIoU /= float(world_size)

        train_class_acc = torch.sum(
            torch.tensor(total_correct_class) / (torch.tensor(total_seen_class).float() + 1e-6)).to(rank)
        train_class_acc = train_class_acc / args.num_existing_type
        torch.distributed.all_reduce(train_class_acc)
        train_class_acc /= float(world_size)

        torch.distributed.all_reduce(loss_sum)
        loss_sum /= float(world_size)
        train_loss = loss_sum / num_train_batch

        # total_correct = torch.tensor(total_correct)
        total_seen = torch.tensor(total_seen).to(rank)
        torch.distributed.all_reduce(total_correct)
        torch.distributed.all_reduce(total_seen)
        train_point_acc = total_correct / float(total_seen)

        if rank == 0:
            save_tensorboard_log(IoUs, epoch, log_writer, mIoU, opt, train_loss, train_point_acc, train_class_acc,
                                 'train_pipeline')

        # ******************* #
        # lr step
        # ******************* #
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        elif args.scheduler == 'cos_warmupLR':
            scheduler.step()
        else:
            raise NotImplemented

        # ******************* #
        # valid
        # ******************* #
        if rank == 0:
            print('-----valid-----')
        with torch.no_grad():
            valid_sampler.set_epoch(epoch)
            model.eval()

            total_correct = 0
            total_seen = 0

            loss_sum = 0
            total_seen_class = [0 for _ in range(args.num_segmentation_type)]
            total_correct_class = [0 for _ in range(args.num_segmentation_type)]
            total_iou_deno_class = [0 for _ in range(args.num_segmentation_type)]

            if rank == 0:
                tqdm_structure = tqdm(enumerate(validation_loader), total=len(validation_loader), smoothing=0.9)
            else:
                tqdm_structure = enumerate(validation_loader)
            for i, (points, target) in tqdm_structure:

                points, target = points.to(rank, non_blocking=True), target.to(rank, non_blocking=True)
                points = models.attention.normalize_data(points)
                points, _ = models.attention.rotate_per_batch(points, None)
                points = points.permute(0, 2, 1)
                batch_size = points.size()[0]

                seg_pred, trans = model(points.float())

                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                batch_label = target.view(-1, 1)[:, 0].data  # array(batch_size*num_points)

                if args.model_para.model == 'pct':
                    loss = criterion(seg_pred.view(-1, args.num_segmentation_type), target.view(-1, 1).squeeze(),
                                     weights, using_weight=args.use_class_weight)  # a scalar
                    if not args.stn_loss_weight == 0:
                        loss = loss + util.feature_transform_reguliarzer(trans) * args.stn_loss_weight

                elif args.model_para.model == 'pointnet':
                    if args.use_class_weight == 0:
                        weights = None
                    loss = criterion(seg_pred.view(-1, args.num_segmentation_type), target.view(-1, 1).squeeze(), trans,
                                     weights)
                elif args.model_para.model == 'pointnet2':
                    if args.use_class_weight == 0:
                        weights = None
                    loss = criterion(seg_pred.view(-1, args.num_segmentation_type),
                                     target.view(-1, 1).squeeze(),
                                     weights)
                elif args.model_para.model == 'dgcnn':
                    if args.use_class_weight == 0:
                        weights = None
                    loss = criterion(seg_pred.view(-1, args.num_segmentation_type),
                                     target.view(-1, 1).squeeze(), weights)

                else:
                    raise NotImplemented

                seg_pred = seg_pred.contiguous().view(-1, args.num_segmentation_type)
                pred_choice = seg_pred.data.max(1)[1]  # array(batch_size*num_points)
                correct = torch.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (batch_size * args.npoints)
                loss_sum += loss

                for l in range(args.num_segmentation_type):
                    total_seen_class[l] += torch.sum((batch_label == l))
                    total_correct_class[l] += torch.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class[l] += torch.sum(((pred_choice == l) | (batch_label == l)))

            IoUs = (torch.Tensor(total_correct_class) / (torch.Tensor(total_iou_deno_class).float() + 1e-6)).to(
                rank)
            mIoU = IoUs.sum() / args.num_existing_type  # torch.mean(IoUs)
            torch.distributed.all_reduce(IoUs)
            torch.distributed.all_reduce(mIoU)
            IoUs /= float(world_size)
            mIoU /= float(world_size)

            eval_class_acc = torch.sum(
                torch.tensor(total_correct_class) / (torch.tensor(total_seen_class).float() + 1e-6)).to(rank)
            eval_class_acc = eval_class_acc / args.num_existing_type
            torch.distributed.all_reduce(eval_class_acc)
            eval_class_acc /= float(world_size)

            torch.distributed.all_reduce(loss_sum)
            loss_sum /= float(world_size)
            valid_loss = loss_sum / num_valid_batch

            # total_correct = torch.tensor(total_correct)
            total_seen = torch.tensor(total_seen).to(rank)
            torch.distributed.all_reduce(total_correct)
            torch.distributed.all_reduce(total_seen)
            valid_point_acc = total_correct / float(total_seen)

            if rank == 0:
                save_tensorboard_log(IoUs, epoch, log_writer, mIoU, opt, valid_loss, valid_point_acc,
                                     eval_class_acc, 'valid_pipeline')

            if (world_size > 1 and rank == 1) or (world_size == 1 and rank == 0):

                if mIoU >= best_mIoU:
                    best_mIoU = mIoU

                    state = {'epoch': epoch,
                             'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': opt.state_dict(),
                             'mIoU': mIoU
                             }

                    if args.finetune == 1:
                        savepath = checkpoints_direction + str(mIoU.item()) + '_best_finetune.pth'
                    else:
                        # savepath = '/home/ies/fu/codes/large_motor_segmentation/best_m.pth'
                        # print('Saving best model at %s' % savepath)
                        # torch.save(state, savepath)

                        savepath = checkpoints_direction + str(mIoU.item()) + '_best_m.pth'

                    print('Saving best model at %s' % savepath)
                    torch.save(state, savepath)

    dist.destroy_process_group()


def train_ddp_func(train_txt,
                   config_dir,
                   valid_motors=None,
                   local_training=False,
                   direction=None):
    args, random_seed, is_local, save_direction, train_dataset, valid_dataset = init_training(train_txt,
                                                                                              config_dir=config_dir,
                                                                                              valid_motors=valid_motors,
                                                                                              local_training=local_training,
                                                                                              direction=direction)

    start_time = time.time()
    world_size = torch.cuda.device_count()

    mp.spawn(train_ddp,
             args=(world_size, args, random_seed, is_local, save_direction, train_dataset, valid_dataset, start_time,
                   local_training),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    pass
