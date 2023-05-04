import os
import datetime
import time
from torch.utils.data import DataLoader
import platform
import numpy as np
import torch
from torch import nn
import shutil

from data_preprocess.data_loader import MotorDataset
from utilities.config import get_parser
from model.pct import PCT_semseg


def train():
    # ******************* #
    # local or server?
    # ******************* #
    system_type = platform.system().lower()  # 'windows' or 'linux'
    is_local = True if system_type == 'windows' else False

    # ******************* #
    # load arguments
    # ******************* #
    args = get_parser()

    # ******************* #
    # make directions
    # ******************* #
    direction = 'outputs/' + str(datetime.date.today()) + '/' + args.train_stamp
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
    data_set_direction = 'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_4debug' if is_local else args.data_dir
    print("start loading training data ...")
    train_dataset = MotorDataset(mode='train',
                                 data_dir=data_set_direction,
                                 num_class=args.num_segmentation_type, num_points=args.npoints,  # 4096
                                 test_area='Validation', sample_rate=1.0)
    print("start loading test data ...")
    validation_set = MotorDataset(mode='valid',
                                  data_dir=data_set_direction,
                                  num_class=args.num_segmentation_type, num_points=args.npoints,  # 4096
                                  test_area='Validation', sample_rate=1.0)

    para_workers = 0 if is_local else 8
    train_loader = DataLoader(train_dataset, num_workers=para_workers, batch_size=args.train_batch_size, shuffle=True,
                              drop_last=True,
                              worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    validation_loader = DataLoader(validation_set, num_workers=para_workers, batch_size=args.test_batch_size,
                                   shuffle=True,
                                   drop_last=False)

    # ******************* #
    # load ML model
    # ******************* #
    device = torch.device("cuda")
    model = PCT_semseg(args).to(device)
    model = nn.DataParallel(model)
    print("use", torch.cuda.device_count(), "GPUs for training")

    # ******************* #
    # opt
    # ******************* #
    if args.use_sgd:
        opt = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=1e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, 0.1, args.epochs)

    # ******************* #
    ## if finetune is true, the the best.pth will be cosidered first
    # ******************* #
    if args.finetune:
        if os.path.exists(direction + '/checkpoints/best.pth'):
            checkpoint = torch.load(direction + '/checkpoints/best.pth')
            print('Use pretrain finetune model to finetune')
        else:
            print('no exiting pretrained model to finetune')
            exit(-1)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])

    else:
        start_epoch = 0


if __name__ == '__main__':
    train()
