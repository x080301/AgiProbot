import os
import datetime
import time
from torch.utils.data import DataLoader
import platform
import numpy as np
import torch
from torch import nn
import shutil
from tqdm import tqdm

from data_preprocess.data_loader import MotorDataset
from utilities.config import get_parser
from model.pct import PCT_semseg
from utilities import util


class BinarySegmentation:
    def __init__(self):

        # ******************* #
        # local or server?
        # ******************* #
        system_type = platform.system().lower()  # 'windows' or 'linux'
        self.is_local = True if system_type == 'windows' else False

        # ******************* #
        # load arguments
        # ******************* #
        self.args = get_parser()
        self.device = torch.device("cuda")

        # ******************* #
        # load ML model
        # ******************* #
        self.model = PCT_semseg(self.args).to(self.device)
        self.model = nn.DataParallel(self.model)
        print("use", torch.cuda.device_count(), "GPUs for training")

    def init_training(self):
        # ******************* #
        # make directions
        # ******************* #
        if self.is_local:
            direction = 'outputs/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') + '/' + self.args.train_stamp
        else:
            direction = '/data/users/fu/' + self.args.titel + '_outputs/' + datetime.datetime.now().strftime(
                '%Y_%m_%d_%H_%M') + '/' + self.args.train_stamp
        if not os.path.exists(direction + '/checkpoints'):
            os.makedirs(direction + '/checkpoints')
        if not os.path.exists(direction + '/train_log'):
            os.makedirs(direction + '/train_log')
        self.checkpoints_direction = direction + '/checkpoints/'

        # ******************* #
        # save mode and parameters
        # ******************* #
        files_to_save = ['train.py', 'config/binary_segmentation.yaml', 'model/pct.py',
                         'data_preprocess/data_loader.py']
        for file_name in files_to_save:
            shutil.copyfile(file_name, direction + '/train_log/' + file_name.split('/')[-1])

        # ******************* #
        # load data set
        # ******************* #
        if self.is_local:
            data_set_direction = 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_4debug'
        else:
            data_set_direction = self.args.data_dir

        print("start loading training data ...")
        train_dataset = MotorDataset(mode='train',
                                     data_dir=data_set_direction,
                                     num_class=self.args.num_segmentation_type, num_points=self.args.npoints,  # 4096
                                     test_area='Validation', sample_rate=1.0)
        print("start loading test data ...")
        validation_set = MotorDataset(mode='valid',
                                      data_dir=data_set_direction,
                                      num_class=self.args.num_segmentation_type, num_points=self.args.npoints,  # 4096
                                      test_area='Validation', sample_rate=1.0)

        para_workers = 0 if self.is_local else 8
        self.train_loader = DataLoader(train_dataset, num_workers=para_workers, batch_size=self.args.train_batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
        self.num_train_batch = len(self.train_loader)
        self.validation_loader = DataLoader(validation_set, num_workers=para_workers,
                                            batch_size=self.args.test_batch_size,
                                            shuffle=True,
                                            drop_last=True)
        self.num_valid_batch = len(self.validation_loader)

        # ******************* #
        # opt
        # ******************* #
        if self.args.use_sgd:
            self.opt = torch.optim.SGD([{'params': self.model.parameters(), 'initial_lr': self.args.lr}], lr=self.args.lr,
                                       momentum=self.args.momentum, weight_decay=1e-4)
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        if self.args.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.epochs, eta_min=1e-5)
        elif self.args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 20, 0.1, self.args.epochs)

        # ******************* #
        # if finetune is true, the the best.pth will be cosidered first
        # ******************* #
        if self.args.finetune:
            if os.path.exists(direction + '/checkpoints/best.pth'):
                checkpoint = torch.load(direction + '/checkpoints/best.pth')
                print('Use pretrain finetune model to finetune')
            else:
                print('no exiting pretrained model to finetune')
                exit(-1)

            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])

        else:
            self.start_epoch = 0

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
        if not self.args.use_class_weight:
            for i in range(self.args.num_segmentation_type):
                weights[i] = 1
        self.weights = weights

        self.best_iou = 0

    def train_epoch(self, epoch):

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
            points, target = points.to(self.device), target.to(self.device)
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

        mIoU__ = np.mean(
            np.array(total_correct_class__) / (np.array(total_iou_deno_class__, dtype=np.float64) + 1e-6))

        if self.args.scheduler == 'cos':
            self.scheduler.step()
        elif self.args.scheduler == 'step':
            if self.opt.param_groups[0]['lr'] > 1e-5:
                self.scheduler.step()
            if self.opt.param_groups[0]['lr'] < 1e-5:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = 1e-5

        print('Epoch %d, train loss: %.6f, train point acc: %.6f ' % (
            epoch, loss_sum / self.num_train_batch, total_correct / float(total_seen)))
        print('Train mean ioU %.6f' % mIoU__)

    def valid_and_save_epoch(self, epoch):
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

            print('-----valid-----')
            for i, (points, seg) in tqdm(enumerate(self.validation_loader), total=len(self.validation_loader),
                                         smoothing=0.9):
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

            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))

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
                         'optimizer_state_dict': self.opt.state_dict(), }
                print('Saving best model at %s' % savepath)
                torch.save(state, savepath)

            print('\n')

    def train(self):

        self.init_training()

        end_epoch = 2 if self.is_local else self.args.epochs

        print('train %d epochs' % (end_epoch - self.start_epoch))
        for epoch in range(self.start_epoch, end_epoch):
            self.train_epoch(epoch)
            self.valid_and_save_epoch(epoch)


if __name__ == '__main__':
    binarysegmentation = BinarySegmentation()
    binarysegmentation.train()
