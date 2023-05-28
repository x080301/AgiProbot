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
import numpy as np

from utilities.config import get_parser
from model.pct import PCTSeg
from data_preprocess.data_loader import MotorDataset
from utilities.lr_scheduler import CosineAnnealingWithWarmupLR
from utilities import util


class BinarySegmentationDPP:
    files_to_save = ['config', 'data_preprocess', 'ideas', 'model', 'train_and_test', 'train_line', 'utilities',
                     'train.py', 'train_line.py']

    def __init__(self, config_dir='config/binary_segmentation.yaml'):
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
            self.data_set_direction = 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_4debug'
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

    def train(self, rank, world_size):
        # ******************* #
        # dpp
        # ******************* #
        torch.manual_seed(self.random_seed)

        backend = 'gloo' if self.is_local else 'nccl'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        if rank == 0:
            log_writer = SummaryWriter(self.save_direction + '/tensorboard_log')

        # ******************* #
        # load ML model
        # ******************* #
        model = PCTSeg(self.args).cuda(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # self.model = nn.DataParallel(self.model)

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
        # if fine tune is true, the the best.pth will be loaded first
        # ******************* #
        if self.args.finetune:
            if os.path.exists('best_m.pth'):
                checkpoint = torch.load('best_m.pth')
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

            model.load_state_dict(checkpoint['model_state_dict'])

        else:
            start_epoch = 0
            end_epoch = 2 if self.is_local else self.args.epochs

        # ******************* #
        # loss function and weights
        # ******************* #
        criterion = util.cal_loss

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

        print(rank)

        if rank == 0:
            print('train %d epochs' % (end_epoch - start_epoch))
        for epoch in range(start_epoch, end_epoch):

            # ******************* #
            # train
            # ******************* #
            if rank == 0:
                print('-----train-----')
            train_sampler.set_epoch(epoch)
            model = model.train()

            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_correct_class__ = [0 for _ in range(self.args.num_segmentation_type)]
            total_iou_deno_class__ = [0 for _ in range(self.args.num_segmentation_type)]

            if rank == 0:
                tqdm_structure = tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9)
            else:
                tqdm_structure = enumerate(train_loader)
            for i, (points, target) in tqdm_structure:

                # ******************* #
                # forwards
                # ******************* #
                points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)
                points = util.normalize_data(points)

                if self.args.after_stn_as_kernel_neighbor_query:  # [bs,4096,3]
                    points, _ = util.rotate_per_batch(points, None)
                else:
                    points, _ = util.rotate_per_batch(points, None)

                points = points.permute(0, 2, 1).float()
                batch_size = points.size()[0]
                opt.zero_grad()

                seg_pred, trans = model(points.float())
                # print(seg_pred)

                # ******************* #
                # backwards
                # ******************* #
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()  # (batch_size,num_points, class_categories)

                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                loss = criterion(seg_pred.view(-1, self.args.num_segmentation_type), target.view(-1, 1).squeeze(),
                                 weights, using_weight=self.args.use_class_weight)  # a scalar

                # loss = loss + util.feature_transform_reguliarzer(trans) * self.args.stn_loss_weight
                loss.backward()
                opt.step()

                # ******************* #
                # further calculation
                # ******************* #
                seg_pred = seg_pred.contiguous().view(-1, self.args.num_segmentation_type)
                # _                                                      (batch_size*num_points , num_class)
                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  # array(batch_size*num_points)
                correct = np.sum(pred_choice == batch_label)
                # when a np arraies have same shape, a==b means in conrrespondending position it equals to one,when they are identical
                total_correct += correct
                total_seen += (batch_size * self.args.npoints)
                loss_sum += loss
                for l in range(self.args.num_segmentation_type):
                    total_correct_class__[l] += np.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class__[l] += np.sum(((pred_choice == l) | (batch_label == l)))

            IoUs = torch.Tensor(total_correct_class__) / (torch.Tensor(total_iou_deno_class__).float() + 1e-6)
            mIoU = torch.mean(IoUs)
            torch.distributed.all_reduce(IoUs)
            torch.distributed.all_reduce(mIoU)
            IoUs /= float(world_size)
            mIoU /= float(world_size)

            torch.distributed.all_reduce(loss_sum)
            loss_sum /= float(world_size)
            train_loss = loss_sum / num_train_batch

            total_correct = torch.tensor(total_correct)
            total_seen = torch.tensor(total_seen)
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
                '''valid_sampler.set_epoch(epoch)
                model = model.eval()

                total_correct = 0
                total_seen = 0

                loss_sum = 0
                labelweights = np.zeros(self.args.num_segmentation_type)
                total_seen_class = [0 for _ in range(self.args.num_segmentation_type)]
                total_correct_class = [0 for _ in range(self.args.num_segmentation_type)]
                total_iou_deno_class = [0 for _ in range(self.args.num_segmentation_type)]

                if rank == 0:
                    tqdm_structure = tqdm(enumerate(validation_loader), total=len(validation_loader), smoothing=0.9)
                else:
                    tqdm_structure = enumerate(validation_loader)
                for i, (points, seg) in tqdm_structure:
                    
                    points, seg = points.to(self.device), seg.to(self.device)
                    points = util.normalize_data(points)
                    points, _ = util.rotate_per_batch(points, None)
                    points = points.permute(0, 2, 1)
                    batch_size = points.size()[0]

                    seg_pred, trans = self.model(points.float())

                    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                    batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()  # array(batch_size*num_points)
                    loss = self.criterion(seg_pred.view(-1, self.args.num_segmentation_type), seg.view(-1, 1).squeeze(),
                                          self.weights, using_weight=self.args.use_class_weight)  # a scalar
                    loss = loss + util.feature_transform_reguliarzer(trans) * self.args.stn_loss_weight
                    seg_pred = seg_pred.contiguous().view(-1, self.args.num_segmentation_type)
                    pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  # array(batch_size*num_points)
                    correct = np.sum(pred_choice == batch_label)
                    total_correct += correct
                    total_seen += (batch_size * self.args.npoints)
                    loss_sum += loss
                    tmp, _ = np.histogram(batch_label, range(self.args.num_segmentation_type + 1))
                    labelweights += tmp

                    for l in range(self.args.num_segmentation_type):
                        total_seen_class[l] += np.sum((batch_label == l))
                        total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                        total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))'''

    def train_dpp(self):
        world_size = torch.cuda.device_count()
        mp.spawn(self.train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    pass
