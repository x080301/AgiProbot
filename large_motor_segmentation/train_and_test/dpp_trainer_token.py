import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import platform
import time
import datetime
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

import models.attention
import utilities.loss_calculation
from utilities.config import get_parser
from models.pct_token import PCTToken
from data_preprocess.data_loader import MotorDataset
from utilities.lr_scheduler import CosineAnnealingWithWarmupLR
from utilities import util


class SegmentationDPP:
    files_to_save = ['config', 'data_preprocess', 'ideas', 'model', 'train_and_test', 'utilities',
                     'train.py', 'train_line.py', 'best_m.pth']

    def __init__(self, train_txt, config_dir='config/binary_segmentation.yaml'):
        # ******************* #
        # load arguments
        # ******************* #
        self.config_dir = config_dir
        self.args = get_parser(config_dir=self.config_dir)
        print("use", torch.cuda.device_count(), "GPUs for training")

        if self.args.random_seed == 0:
            self.random_seed = int(time.time())
        else:
            self.random_seed = self.args.train.random_seed
        # ******************* #
        # local or server?
        # ******************* #
        system_type = platform.system().lower()  # 'windows' or 'linux'
        self.is_local = True if system_type == 'windows' else False
        if self.is_local:
            self.args.npoints = 1024
            self.args.sample_rate = 1.
            self.args.ddp.gpus = 1
            if self.args.finetune == 0:
                self.data_set_direction = 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_4debug'
            else:
                self.data_set_direction = 'E:/datasets/agiprobot/binary_label/big_motor_zivid_binlabel_npy_for_debug'
            # 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_npy'
        else:
            self.data_set_direction = self.args.data_dir

        # ******************* #
        # make directions
        # ******************* #
        if self.is_local:
            direction = 'outputs/' + self.args.titel + '_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        else:
            direction = '/data/users/fu/' + self.args.titel + '_outputs/' + \
                        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        if not os.path.exists(direction + '/checkpoints'):
            os.makedirs(direction + '/checkpoints')
        if not os.path.exists(direction + '/train_log'):
            os.makedirs(direction + '/train_log')
        if not os.path.exists(direction + '/tensorboard_log'):
            os.makedirs(direction + '/tensorboard_log')
        self.checkpoints_direction = direction + '/checkpoints/'

        # ******************* #
        # save mode and parameters
        # ******************* #
        if train_txt is not None:
            print(train_txt)
            shutil.copyfile(train_txt, direction + '/train_log/' + train_txt.split('/')[-1])
        for file_name in self.files_to_save:
            if '.' in file_name:
                shutil.copyfile(file_name, direction + '/train_log/' + file_name.split('/')[-1])
            else:
                shutil.copytree(file_name, direction + '/train_log/' + file_name.split('/')[-1])

        with open(direction + '/train_log/' + 'random_seed_' + str(self.random_seed) + '.txt', 'w') as f:
            f.write('')

        self.save_direction = direction

        # ******************* #
        # load data set
        # ******************* #
        print("start loading training data ...")

        self.train_dataset = MotorDataset(mode='train',
                                          data_dir=self.data_set_direction,
                                          num_class=self.args.num_segmentation_type, num_points=self.args.npoints,
                                          # 4096
                                          test_area='Validation', sample_rate=self.args.sample_rate)
        print("start loading test data ...")
        self.valid_dataset = MotorDataset(mode='valid',
                                          data_dir=self.data_set_direction,
                                          num_class=self.args.num_segmentation_type, num_points=self.args.npoints,
                                          # 4096
                                          test_area='Validation', sample_rate=1.0)

        # ******************* #
        # dpp
        # ******************* #
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # ******************* #
        # load ML model
        # ******************* #

        self.model = PCTToken(self.args)

        # if fine tune is true, the the best.pth will be loaded first

        if self.args.finetune == 1:
            if os.path.exists(self.args.pretrained_model_path):
                checkpoint = torch.load(self.args.pretrained_model_path)
                print('Use pretrained model for fine tune')
            else:
                print('no exiting pretrained model')
                exit(-1)

            start_epoch = checkpoint['epoch']

            end_epoch = start_epoch
            end_epoch += 2 if self.is_local else self.args.epochs

            if 'mIoU' in checkpoint:
                print('train begin at %dth epoch with mIoU %.6f' % (start_epoch, checkpoint['mIoU']))
            else:
                print('train begin with %dth epoch' % start_epoch)

            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k[0:7] == 'module.':
                    new_state_dict[k[7:]] = v
                else:
                    break
            self.model.load_state_dict(new_state_dict)  # .to(rank)
        else:
            start_epoch = 0
            end_epoch = 2 if self.is_local else self.args.epochs

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        # self.model = nn.DataParallel(self.model)

    def train(self, rank, world_size):
        best_mIoU = 0
        # ******************* #
        # dpp and load ML model
        # ******************* #
        torch.cuda.set_device(rank)
        backend = 'gloo' if self.is_local else 'nccl'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        start_epoch = self.start_epoch
        end_epoch = self.end_epoch

        model = copy.deepcopy(self.model)
        model.to(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        torch.manual_seed(self.random_seed)

        if rank == 0:
            log_writer = SummaryWriter(self.save_direction + '/tensorboard_log')

        # ******************* #
        # load dataset
        # ******************* #
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,
                                                                        num_replicas=self.args.ddp.world_size,
                                                                        rank=rank
                                                                        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset,
                                                                        num_replicas=self.args.ddp.world_size,
                                                                        rank=rank
                                                                        )

        para_workers = 0 if self.is_local else 8
        train_loader = DataLoader(self.train_dataset,
                                  num_workers=para_workers,
                                  batch_size=self.args.train_batch_size,
                                  shuffle=False,
                                  drop_last=True,
                                  pin_memory=True,
                                  sampler=train_sampler
                                  )
        num_train_batch = len(train_loader)

        validation_loader = DataLoader(self.valid_dataset,
                                       num_workers=para_workers,
                                       pin_memory=True,
                                       sampler=valid_sampler,
                                       batch_size=self.args.test_batch_size,
                                       shuffle=False,
                                       drop_last=True
                                       )
        num_valid_batch = len(validation_loader)

        # ******************* #
        # opt
        # ******************* #
        if self.args.use_sgd:
            opt = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': self.args.lr}],
                                  lr=self.args.lr,
                                  momentum=self.args.momentum, weight_decay=1e-4)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        if self.args.scheduler == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.args.epochs,
                                                                   eta_min=self.args.end_lr)
        elif self.args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, 0.1, self.args.epochs)
        elif self.args.scheduler == 'cos_warmupLR':
            scheduler = CosineAnnealingWithWarmupLR(opt,
                                                    T_max=self.args.epochs - self.args.cos_warmupLR.warmup_epochs,
                                                    eta_min=self.args.end_lr,
                                                    warmup_init_lr=self.args.end_lr,
                                                    warmup_epochs=self.args.cos_warmupLR.warmup_epochs)

        else:
            print('no scheduler called' + self.args.scheduler)
            exit(-1)

        # ******************* #
        # loss function and weights
        # ******************* #

        weights = torch.Tensor(self.train_dataset.label_weights).cuda()
        # print(weights)
        percentage = torch.Tensor(self.train_dataset.persentage_each_type).cuda()
        scale = weights * percentage
        scale = 1 / scale.sum()
        # print(scale)
        weights *= scale
        # print(weights)
        if self.args.use_class_weight == 0:
            for i in range(self.args.num_segmentation_type):
                weights[i] = 1

        if rank == 0:
            print('train %d epochs' % (end_epoch - start_epoch))
        for epoch in range(start_epoch, end_epoch):

            # ******************* #
            # train
            # ******************* #
            if rank == 0:
                print('-----train-----')
            train_sampler.set_epoch(epoch)
            model.train()

            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_correct_class__ = [0 for _ in range(self.args.num_segmentation_type)]
            total_iou_deno_class__ = [0 for _ in range(self.args.num_segmentation_type)]

            if rank == 0 and epoch == start_epoch:
                tqdm_structure = tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9)
            else:
                tqdm_structure = enumerate(train_loader)
            for i, (points, target, token_) in tqdm_structure:
                '''
                points: (B,N,3)
                target: (B,N)
                '''
                # ******************* #
                # forwards
                # ******************* #
                target = target.cuda(non_blocking=True)

                points = points.cuda(non_blocking=True)
                points = models.attention.normalize_data(points, )

                # rotation augmentation
                points, _ = models.attention.rotate_per_batch(points, None)

                points = points.permute(0, 2, 1).float()
                batch_size = points.size()[0]
                opt.zero_grad()

                point_segmentation_pred, \
                bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals, \
                transform_matrix = model(points.float())
                # (B, segment_type, N)
                # (B,2,T), (B,bolt_type,T), (B,3,T), (B,3,T)
                # (B,3,3)

                # ******************* #
                # backwards
                # ******************* #
                # loss
                loss = utilities.loss_calculation.cal_loss(point_segmentation_pred, target, weights, transform_matrix,
                                                           bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals,
                                                           self.args)

                loss.backward()
                opt.step()

                # ******************* #
                # further calculation
                # ******************* #

                batch_label = target.view(-1, 1)[:, 0]

                point_segmentation_pred = point_segmentation_pred.permute(0, 2, 1)
                point_segmentation_pred = point_segmentation_pred.contiguous().view(-1, self.args.num_segmentation_type)
                # _                                                      (batch_size*num_points , num_class)
                pred_choice = point_segmentation_pred.data.max(1)[1]  # array(batch_size*num_points)

                correct = torch.sum(pred_choice == batch_label)
                # when a np arraies have same shape, a==b means in conrrespondending position it equals to one,when they are identical
                total_correct += correct
                total_seen += (batch_size * self.args.npoints)
                loss_sum += loss
                for l in range(self.args.num_segmentation_type):
                    total_correct_class__[l] += torch.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class__[l] += torch.sum(((pred_choice == l) | (batch_label == l)))

            IoUs = (torch.tensor(total_correct_class__) / (torch.tensor(total_iou_deno_class__).float() + 1e-6)).to(
                rank)
            mIoU = torch.mean(IoUs)
            torch.distributed.all_reduce(IoUs)
            torch.distributed.all_reduce(mIoU)
            IoUs /= float(world_size)
            mIoU /= float(world_size)

            torch.distributed.all_reduce(loss_sum)
            loss_sum /= float(world_size)
            train_loss = loss_sum / num_train_batch

            # total_correct = torch.tensor(total_correct)
            total_seen = torch.tensor(total_seen).to(rank)
            torch.distributed.all_reduce(total_correct)
            torch.distributed.all_reduce(total_seen)
            train_point_acc = total_correct / float(total_seen)

            if rank == 0:
                log_writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)
                log_writer.add_scalar('IoU_background/train_IoU_background', IoUs[0], epoch)
                log_writer.add_scalar('IoU_motor/train_IoU_motor', IoUs[1], epoch)
                log_writer.add_scalar('mIoU/train_mIoU', mIoU, epoch)
                log_writer.add_scalar('loss/train_loss', train_loss, epoch)
                log_writer.add_scalar('point_acc/train_point_acc', train_point_acc, epoch)
                print('Epoch %d, train loss: %.6f, train point acc: %.6f ' % (
                    epoch, train_loss, train_point_acc))
                print('Train mean ioU %.6f' % mIoU)

            # ******************* #
            # lr step
            # ******************* #
            if self.args.scheduler == 'cos':
                scheduler.step()
            elif self.args.scheduler == 'step':
                if opt.param_groups[0]['lr'] > 1e-5:
                    scheduler.step()
                if opt.param_groups[0]['lr'] < 1e-5:
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = 1e-5
            elif self.args.scheduler == 'cos_warmupLR':
                # print(self.opt.param_groups[0]['lr'])
                scheduler.step()

            # ******************* #
            # valid
            # ******************* #
            if rank == 0:
                print('-----valid-----')
            with torch.no_grad():
                pass
                valid_sampler.set_epoch(epoch)
                model.eval()

                total_correct = 0
                total_seen = 0

                loss_sum = 0
                total_seen_class = [0 for _ in range(self.args.num_segmentation_type)]
                total_correct_class = [0 for _ in range(self.args.num_segmentation_type)]
                total_iou_deno_class = [0 for _ in range(self.args.num_segmentation_type)]

                if rank == 0 and epoch == start_epoch:
                    tqdm_structure = tqdm(enumerate(validation_loader), total=len(validation_loader), smoothing=0.9)
                else:
                    tqdm_structure = enumerate(validation_loader)
                for i, (points, target) in tqdm_structure:
                    '''
                    points: (B,N,C)
                    target: (B,N,2+bolt_type+3+3)
                    '''

                    points, target = points.cuda(non_blocking=True), target.cuda(
                        non_blocking=True)
                    points = models.attention.normalize_data(points)
                    points, _ = models.attention.rotate_per_batch(points, None)
                    points = points.permute(0, 2, 1)
                    batch_size = points.size()[0]

                    point_segmentation_pred, \
                    bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals, \
                    transform_matrix = model(points.float())
                    # (B, segment_type, N)
                    # (B,2,T), (B,bolt_type,T), (B,3,T), (B,3,T)
                    # (B,3,3)

                    # loss
                    loss = utilities.loss_calculation.cal_loss(point_segmentation_pred, target, weights, transform_matrix,
                                                               bolt_existing_label, bolt_type_pred, bolt_centers, bolt_normals,
                                                               self.args)
                    loss_sum += loss

                    batch_label = target.view(-1, 1)[:, 0].data

                    point_segmentation_pred = point_segmentation_pred.permute(0, 2, 1).contiguous()
                    point_segmentation_pred = point_segmentation_pred.view(-1, self.args.num_segmentation_type)
                    pred_choice = point_segmentation_pred.data.max(1)[1]  # array(batch_size*num_points)
                    correct = torch.sum(pred_choice == batch_label)
                    total_correct += correct
                    total_seen += (batch_size * self.args.npoints)

                    for l in range(self.args.num_segmentation_type):
                        total_seen_class[l] += torch.sum((batch_label == l))
                        total_correct_class[l] += torch.sum((pred_choice == l) & (batch_label == l))
                        total_iou_deno_class[l] += torch.sum(((pred_choice == l) | (batch_label == l)))

                IoUs = (torch.Tensor(total_correct_class) / (torch.Tensor(total_iou_deno_class).float() + 1e-6)).to(
                    rank)
                mIoU = torch.mean(IoUs)
                torch.distributed.all_reduce(IoUs)
                torch.distributed.all_reduce(mIoU)
                IoUs /= float(world_size)
                mIoU /= float(world_size)

                eval_class_acc = torch.mean(
                    torch.tensor(total_correct_class) / (torch.tensor(total_seen_class).float() + 1e-6)).to(rank)
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
                    log_writer.add_scalar('loss/eval_loss', valid_loss, epoch)
                    log_writer.add_scalar('point_acc/eval_point_acc', valid_point_acc, epoch)
                    log_writer.add_scalar('point_acc/eval_class_acc', eval_class_acc, epoch)
                    log_writer.add_scalar('mIoU/eval_mIoU', mIoU, epoch)
                    log_writer.add_scalar('IoU_background/eval_IoU_background', IoUs[0], epoch)
                    log_writer.add_scalar('IoU_motor/eval_IoU_motor', IoUs[1], epoch)

                    outstr = 'Epoch %d,  eval loss %.6f, eval point acc %.6f, eval avg class acc %.6f' % (
                        epoch, valid_loss,
                        valid_point_acc,
                        eval_class_acc)
                    print(outstr)
                    print('Valid mean ioU %.6f' % mIoU)

                    if mIoU >= best_mIoU:
                        best_mIoU = mIoU

                        state = {'epoch': epoch,
                                 'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': opt.state_dict(),
                                 'mIoU': mIoU
                                 }

                        if self.args.finetune == 1:
                            savepath = self.checkpoints_direction + str(mIoU.item()) + '_best_finetune.pth'
                        else:
                            savepath = '/home/ies/fu/codes/large_motor_segmentation/best_m.pth'
                            print('Saving best model at %s' % savepath)
                            torch.save(state, savepath)

                            savepath = self.checkpoints_direction + str(mIoU.item()) + 'best_m.pth'

                        print('Saving best model at %s' % savepath)
                        torch.save(state, savepath)

                    print('\n')

    def train_dpp(self):
        world_size = torch.cuda.device_count()
        mp.spawn(self.train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    pass
