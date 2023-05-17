import datetime
import os
import platform
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist

from data_preprocess.data_loader import MotorDataset, MotorDatasetTest
from model.pct import PCT_semseg
from utilities import util
from utilities.config import get_parser
from utilities.lr_scheduler import CosineAnnealingWithWarmupLR


class BinarySegmentation:
    # 'config/binary_segmentation.yaml' should be at the end. It can be changed latter.
    files_to_save = ['train.py', 'model/pct.py', 'data_preprocess/data_loader.py', 'config/binary_segmentation.yaml']

    def __init__(self, config_dir='config/binary_segmentation.yaml'):
        self.best_iou = 0

        # ******************* #
        # load arguments
        # ******************* #
        self.config_dir = config_dir
        self.args = get_parser(config_dir=self.config_dir)

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
        if self.is_local:
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
        self.files_to_save[-1] = self.config_dir
        for file_name in self.files_to_save:
            shutil.copyfile(file_name, direction + '/train_log/' + file_name.split('/')[-1])
        self.writer = SummaryWriter(direction + '/tensorboard_log')
        with open(direction + '/train_log/' + 'random_seed_' + str(self.random_seed) + '.txt', 'w') as f:
            f.write('')

    def init_training(self, gpu):
        rank = gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=self.args.ddp.world_size, rank=rank)

        torch.manual_seed(self.random_seed)
        torch.cuda.set_device(gpu)
        # ******************* #
        # load ML model
        # ******************* #
        self.model = PCT_semseg(self.args).cuda(gpu)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[gpu])
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        # self.model = nn.DataParallel(self.model)
        print("use", torch.cuda.device_count(), "GPUs for training")

        # ******************* #
        # load data set
        # ******************* #
        print("start loading training data ...")

        train_dataset = MotorDataset(mode='train',
                                     data_dir=self.data_set_direction,
                                     num_class=self.args.num_segmentation_type, num_points=self.args.npoints,  # 4096
                                     test_area='Validation', sample_rate=self.args.sample_rate)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                             num_replicas=self.args.ddp.world_size,
                                                                             rank=rank
                                                                             )
        print("start loading test data ...")
        valid_dataset = MotorDataset(mode='valid',
                                     data_dir=self.data_set_direction,
                                     num_class=self.args.num_segmentation_type, num_points=self.args.npoints,  # 4096
                                     test_area='Validation', sample_rate=1.0)
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                                             num_replicas=self.args.ddp.world_size,
                                                                             rank=rank
                                                                             )

        # para_workers = 0 if self.is_local else 8
        self.train_loader = DataLoader(train_dataset,
                                       # num_workers=para_workers,
                                       batch_size=self.args.train_batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       # worker_init_fn=lambda x: np.random.seed(x + int(time.time())),  # TODO 是否有影响？
                                       pin_memory=True,
                                       sampler=self.train_sampler
                                       )
        self.num_train_batch = len(self.train_loader)

        self.validation_loader = DataLoader(valid_dataset,
                                            # num_workers=para_workers,
                                            pin_memory=True,
                                            sampler=self.valid_sampler,
                                            batch_size=self.args.test_batch_size,
                                            shuffle=True,
                                            drop_last=True
                                            )
        self.num_valid_batch = len(self.validation_loader)

        # ******************* #
        # opt
        # ******************* #
        if self.args.use_sgd:
            self.opt = torch.optim.SGD([{'params': self.model.parameters(), 'initial_lr': self.args.lr}],
                                       lr=self.args.lr,
                                       momentum=self.args.momentum, weight_decay=1e-4)
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        if self.args.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.args.epochs,
                                                                        eta_min=self.args.end_lr)
        elif self.args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 20, 0.1, self.args.epochs)
        elif self.args.scheduler == 'cos_warmupLR':
            self.scheduler = CosineAnnealingWithWarmupLR(self.opt,
                                                         T_max=self.args.epochs - self.args.cos_warmupLR.warmup_epochs,
                                                         eta_min=self.args.end_lr,
                                                         warmup_init_lr=self.args.end_lr,
                                                         warmup_epochs=self.args.cos_warmupLR.warmup_epochs)

        else:
            print('no scheduler called' + self.args.scheduler)
            exit(-1)

        # ******************* #
        # if finetune is true, the the best.pth will be cosidered first
        # ******************* #
        if self.args.finetune:
            if os.path.exists('best_m.pth'):
                checkpoint = torch.load('best_m.pth')
                print('Use pretrain finetune model to finetune')
            else:
                print('no exiting pretrained model to finetune')
                exit(-1)

            self.start_epoch = checkpoint['epoch']

            self.end_epoch = self.start_epoch
            self.end_epoch += 2 if self.is_local else self.args.epochs

            if 'mIoU' in checkpoint:
                print('train begin at %dth epoch with mIoU %.6f' % (self.start_epoch, checkpoint['mIoU']))
            else:
                print('train begin with %dth epoch' % self.start_epoch)

            self.model.load_state_dict(checkpoint['model_state_dict'])

        else:
            self.start_epoch = 0
            self.end_epoch = 2 if self.is_local else self.args.epochs

        # ******************* #
        # loss function and weights
        # ******************* #
        self.criterion = util.cal_loss

        weights = torch.Tensor(train_dataset.label_weights).cuda()
        # print(weights)
        persentige = torch.Tensor(train_dataset.persentage_each_type).cuda()
        scale = weights * persentige
        scale = 1 / scale.sum()
        # print(scale)
        weights *= scale
        # print(weights)
        if self.args.use_class_weight == 0:
            for i in range(self.args.num_segmentation_type):
                weights[i] = 1
        self.weights = weights

    def train_epoch(self, epoch):

        self.train_sampler.set_epoch(epoch)

        self.model = self.model.train()
        self.args.training = True

        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_correct_class__ = [0 for _ in range(self.args.num_segmentation_type)]
        total_iou_deno_class__ = [0 for _ in range(self.args.num_segmentation_type)]

        print('-----train-----')
        for i, (points, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), smoothing=0.9):

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
            self.opt.zero_grad()

            seg_pred, trans = self.model(points.float())
            # print(seg_pred)

            # ******************* #
            # backwards
            # ******************* #
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()  # (batch_size,num_points, class_categories)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            loss = self.criterion(seg_pred.view(-1, self.args.num_segmentation_type), target.view(-1, 1).squeeze(),
                                  self.weights, using_weight=self.args.use_class_weight)  # a scalar

            loss = loss + util.feature_transform_reguliarzer(trans) * self.args.stn_loss_weight
            loss.backward()
            self.opt.step()

            # ******************* #
            # further calculation
            # ******************* #
            seg_pred = seg_pred.contiguous().view(-1,
                                                  self.args.num_segmentation_type)  # (batch_size*num_points , num_class)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  # array(batch_size*num_points)
            correct = np.sum(
                pred_choice == batch_label)  # when a np arraies have same shape, a==b means in conrrespondending position it equals to one,when they are identical
            total_correct += correct
            total_seen += (batch_size * self.args.npoints)
            loss_sum += loss
            for l in range(self.args.num_segmentation_type):
                total_correct_class__[l] += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class__[l] += np.sum(((pred_choice == l) | (batch_label == l)))

        IoUs = np.array(total_correct_class__) / (np.array(total_iou_deno_class__, dtype=np.float64) + 1e-6)
        mIoU = np.mean(IoUs)

        self.writer.add_scalar('lr', self.opt.param_groups[0]['lr'], epoch)
        self.writer.add_scalar('loss/train_loss', loss_sum / self.num_train_batch, epoch)
        self.writer.add_scalar('point_acc/train_point_acc', total_correct / float(total_seen), epoch)
        self.writer.add_scalar('mIoU/train_mIoU', mIoU, epoch)
        self.writer.add_scalar('IoU_background/train_IoU_background', IoUs[0], epoch)
        self.writer.add_scalar('IoU_motor/train_IoU_motor', IoUs[1], epoch)

        if self.args.scheduler == 'cos':
            self.scheduler.step()
        elif self.args.scheduler == 'step':
            if self.opt.param_groups[0]['lr'] > 1e-5:
                self.scheduler.step()
            if self.opt.param_groups[0]['lr'] < 1e-5:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = 1e-5
        elif self.args.scheduler == 'cos_warmupLR':
            # print(self.opt.param_groups[0]['lr'])
            self.scheduler.step()

        print('Epoch %d, train loss: %.6f, train point acc: %.6f ' % (
            epoch, loss_sum / self.num_train_batch, total_correct / float(total_seen)))
        print('Train mean ioU %.6f' % mIoU)

    def valid_and_save_epoch(self, epoch):
        with torch.no_grad():
            self.valid_sampler.set_epoch(epoch)

            total_correct = 0
            total_seen = 0

            loss_sum = 0
            labelweights = np.zeros(self.args.num_segmentation_type)
            total_seen_class = [0 for _ in range(self.args.num_segmentation_type)]
            total_correct_class = [0 for _ in range(self.args.num_segmentation_type)]
            total_iou_deno_class = [0 for _ in range(self.args.num_segmentation_type)]

            self.model = self.model.eval()
            self.args.training = False

            print('-----valid-----')
            for i, (points, seg) in tqdm(enumerate(self.validation_loader), total=len(self.validation_loader),
                                         smoothing=0.9):
                points, seg = points.cuda(non_blocking=True), seg.cuda(non_blocking=True)
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
                    total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))

            IoUs = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
            mIoU = np.mean(IoUs)

            self.writer.add_scalar('loss/eval_loss', loss_sum / self.num_valid_batch, epoch)
            self.writer.add_scalar('point_acc/eval_point_acc', total_correct / float(total_seen), epoch)
            self.writer.add_scalar('point_acc/eval_class_acc', np.mean(
                np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6)), epoch)
            self.writer.add_scalar('mIoU/eval_mIoU', mIoU, epoch)
            self.writer.add_scalar('IoU_background/eval_IoU_background', IoUs[0], epoch)
            self.writer.add_scalar('IoU_motor/eval_IoU_motor', IoUs[1], epoch)

            outstr = 'Epoch %d,  eval loss %.6f, eval point acc %.6f, eval avg class acc %.6f' % (
                epoch, (loss_sum / self.num_valid_batch),
                (total_correct / float(total_seen)),
                (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
            print(outstr)
            print('Valid mean ioU %.6f' % mIoU)

            if mIoU >= self.best_iou:
                self.best_iou = mIoU
                if self.args.finetune:
                    savepath = self.checkpoints_direction + str(mIoU) + '_best_finetune.pth'
                else:
                    savepath = self.checkpoints_direction + str(mIoU) + 'best_m.pth'

                state = {'epoch': epoch,
                         'model_state_dict': self.model.state_dict(),
                         'optimizer_state_dict': self.opt.state_dict(),
                         'mIoU': mIoU
                         }
                print('Saving best model at %s' % savepath)
                torch.save(state, savepath)

            print('\n')

    def load_trained_model(self, check_points_file_dir):

        checkpoint = torch.load(check_points_file_dir)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def test(self, point_cloud_dir, save_dir=None, save_pcd_dir=None):
        import pandas

        file_dir = []
        mIoU = []
        test_loss = []
        test_point_acc = []
        test_avg_class_acc = []
        for file_name in tqdm(os.listdir(point_cloud_dir)):
            test_data = MotorDatasetTest(point_cloud_dir=point_cloud_dir + '/' + file_name, num_class=2,
                                         num_points=4096)
            test_loader = DataLoader(test_data, num_workers=0,
                                     batch_size=self.args.test_batch_size,
                                     shuffle=True,
                                     drop_last=False)

            with torch.no_grad():
                total_correct = 0
                total_seen = 0

                loss_sum = 0
                labelweights = np.zeros(self.args.num_segmentation_type)
                total_seen_class = [0 for _ in range(self.args.num_segmentation_type)]
                total_correct_class = [0 for _ in range(self.args.num_segmentation_type)]
                total_iou_deno_class = [0 for _ in range(self.args.num_segmentation_type)]

                self.model = self.model.eval()
                self.args.training = False

                for i, (points, seg) in enumerate(test_loader):
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
                        total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))

                file_dir.append(point_cloud_dir + '/' + file_name)
                mIoU.append(np.mean(
                    np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)))
                test_loss.append(loss_sum / self.num_valid_batch)
                test_point_acc.append(total_correct / float(total_seen))
                test_avg_class_acc.append(np.mean(
                    np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6)))

        if save_dir is not None:
            destination_csv_dir = save_dir + '/' + 'results.csv'
            if os.path.isfile(destination_csv_dir):
                print('csv exists. overwrite!')

            data_frame = pandas.DataFrame(
                {'file': file_dir, 'mIoU': mIoU, 'test_loss': test_loss, 'test_point_acc': test_point_acc,
                 'test_avg_class_acc': test_avg_class_acc})
            data_frame.to_csv(destination_csv_dir, index=False)

    def train(self, gpu):

        self.init_training(gpu)
        print('train %d epochs' % (self.end_epoch - self.start_epoch))
        for epoch in range(self.start_epoch, self.end_epoch):
            self.train_epoch(epoch)
            self.valid_and_save_epoch(epoch)


def train_ddp():
    config_dir = 'config/binary_segmentation.yaml'
    print(config_dir)
    binarysegmentation = BinarySegmentation(config_dir=config_dir)
    train = binarysegmentation.train

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    # self.train(0)
    mp.spawn(train, nprocs=1)
