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
import util


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

        # ******************* #
        # make directions
        # ******************* #
        direction = 'outputs/' + str(datetime.date.today()) + '/' + self.args.train_stamp
        if not os.path.exists(direction + '/checkpoints'):
            os.makedirs(direction + '/checkpoints')
        if not os.path.exists(direction + '/train_log'):
            os.makedirs(direction + '/train_log')

        # ******************* #
        # save mode and parameters
        # ******************* #
        files_to_save = ['train.py', 'config/binary_segmentation.yaml', 'model/pct.py']
        for file_name in files_to_save:
            shutil.copyfile(file_name, direction + '/train_log/' + file_name.split('/')[-1])

        # ******************* #
        # load data set
        # ******************* #
        data_set_direction = 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_4debug' if self.is_local else self.args.data_dir
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
        self.validation_loader = DataLoader(validation_set, num_workers=para_workers,
                                            batch_size=self.args.test_batch_size,
                                            shuffle=True,
                                            drop_last=False)

        # ******************* #
        # load ML model
        # ******************* #
        self.device = torch.device("cuda")
        model = PCT_semseg(self.args).to(self.device)
        model = nn.DataParallel(model)
        print("use", torch.cuda.device_count(), "GPUs for training")

        # ******************* #
        # opt
        # ******************* #
        if self.args.use_sgd:
            opt = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': self.args.lr}], lr=self.args.lr,
                                  momentum=self.args.momentum, weight_decay=1e-4)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        if self.args.scheduler == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.args.epochs, eta_min=1e-5)
        elif self.args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, 0.1, self.args.epochs)

        self.opt = opt
        self.scheduler = scheduler

        # ******************* #
        ## if finetune is true, the the best.pth will be cosidered first
        # ******************* #
        if self.args.finetune:
            if os.path.exists(direction + '/checkpoints/best.pth'):
                checkpoint = torch.load(direction + '/checkpoints/best.pth')
                print('Use pretrain finetune model to finetune')
            else:
                print('no exiting pretrained model to finetune')
                exit(-1)

            self.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])

        else:
            self.start_epoch = 0

        # ******************* #
        # loss function and weights
        # ******************* #
        self.criterion = util.cal_loss

        weights = torch.Tensor(train_dataset.label_weights).cuda()
        persentige = torch.Tensor(train_dataset.persentage_each_type).cuda()
        scale = weights * persentige
        scale = 1 / scale.sum()
        weights *= scale
        if self.args.use_class_weight == 0:
            for i in range(self.args.num_segmentation_type):
                weights[i] = 1
        self.weights = weights

        self.model = model

    def train_epoch(self):

        self.model = self.model.train()
        self.args.training = True

        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_correct_class__ = [0 for _ in range(self.args.num_segmentation_type)]
        total_iou_deno_class__ = [0 for _ in range(self.args.num_segmentation_type)]

        for i, (points, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), smoothing=0.9):

            # ******************* #
            # forwards
            # ******************* #
            points, target = points.to(self.device), target.to(self.device)
            points = util.normalize_data(points)

            if self.args.after_stn_as_kernel_neighbor_query:  # [bs,4096,3]
                points, _, GT = util.rotate_per_batch(points, None)  # TODO What's GT?
            else:
                points, _, GT = util.rotate_per_batch(points, None)

            points = points.permute(0, 2, 1).float()
            batch_size = points.size()[0]
            self.opt.zero_grad()

            seg_pred, trans = self.model(points.float())

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

        outstr = 'Segmentation:Train %d, loss: %.6f, train acc: %.6f ' % (
            epoch, (loss_sum / num_batches), (total_correct / float(total_seen)))
        outstr = 'Classification:Train %d, loss: %.6f, train acc: %.6f ' % (
            epoch, (loss_class / num_batches), (total_correct_classification / float(total_seen_classification)))
        io.cprint(outstr)
        writer.add_scalar('Training loss', (loss_sum / num_batches), epoch)
        writer.add_scalar('Training accuracy', (total_correct / float(total_seen)), epoch)
        writer.add_scalar('Training mean ioU', (mIoU__), epoch)
        writer.add_scalar('Training loU of bolt', (total_correct_class__[5] / float(total_iou_deno_class__[5])),
                          epoch)

    def train(self):

        end_epoch = 3 if self.is_local else self.args.epochs
        for epoch in range(self.start_epoch, end_epoch):
            self.train_epoch()


if __name__ == '__main__':
    binarysegmentation = BinarySegmentation()
    binarysegmentation.train()
