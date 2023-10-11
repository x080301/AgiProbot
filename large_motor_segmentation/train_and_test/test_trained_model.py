import torch

import models.attention
import utilities.loss_calculation
from utilities.config import get_parser
from torch.utils.data import DataLoader
from data_preprocess.data_loader import MotorDatasetTest
from tqdm import tqdm
import einops
import numpy as np
import os

from utilities import util

label_rgb_dic = {
    'Gear': [102, 140, 255],
    'Connector': [102, 255, 102],
    'bolt': [247, 77, 77],
    'Solenoid': [255, 165, 0],
    'Electrical Connector': [255, 255, 0],
    'Main Housing': [0, 100, 0]
}


def save_pcd(points, labels, save_name):
    import open3d as o3d

    points, labels = points.cpu(), labels.cpu()

    colors = torch.asarray(list(label_rgb_dic.values())) / 255.0
    colors = torch.index_select(colors, dim=0, index=labels)

    # keys = list(label_rgb_dic.keys())
    #
    # colors = torch.zeros((0, 3))
    # for label in tqdm(labels):
    #     colors = torch.cat((colors, torch.asarray(label_rgb_dic[keys[label]]).resize(1, 3)), dim=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.numpy())

    # pcd.paint_uniform_color([0, 1, 0])

    o3d.io.write_point_cloud(save_name, pcd, write_ascii=True)


class TestTrainedModel():
    def __init__(self, config_dir, model_name, data_set_dir, check_point_dir, pcd_save_dir=None, plt_save_dir=None):

        self.args = get_parser(config_dir=config_dir)

        if model_name == 'pct pipeline':
            from models.pct import PCTPipeline
            self.model = PCTPipeline(self.args)
        elif model_name == 'pct token':
            from models.pct import PCTToken
            self.model = PCTToken(self.args)
        else:
            raise Exception("Not implemented")

        checkpoint = torch.load(check_point_dir)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k[0:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.cuda()

        test_dataset = MotorDatasetTest(data_set_dir, self.args.num_segmentation_type, self.args.npoints)
        self.test_data_loader = DataLoader(test_dataset,
                                           batch_size=self.args.test_batch_size,
                                           shuffle=True,
                                           drop_last=False)

        self.save_dir = pcd_save_dir
        self.plt_save_dir = plt_save_dir

        self.criterion = utilities.loss_calculation.cal_loss

    def run_test(self):
        with torch.no_grad():

            self.model.eval()

            total_correct = 0
            total_seen = 0

            loss_sum = 0
            total_seen_class = [0 for _ in range(self.args.num_segmentation_type)]
            total_correct_class = [0 for _ in range(self.args.num_segmentation_type)]
            total_iou_deno_class = [0 for _ in range(self.args.num_segmentation_type)]

            num_test_batch = len(self.test_data_loader)

            class_counts = torch.zeros(self.args.num_segmentation_type, self.args.num_segmentation_type)

            all_predictions = None
            for i, (points, labels, redundant_data_mask) in tqdm(enumerate(self.test_data_loader),
                                                                 total=len(self.test_data_loader),
                                                                 smoothing=0.9):
                raw_points = points
                points, labels = points.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                points = models.attention.normalize_data(points)
                points, _ = models.attention.rotate_per_batch(points, None)
                points = points.permute(0, 2, 1)
                batch_size = points.size()[0]

                seg_pred, trans = self.model(points.float())  # input (B,3,N)

                seg_pred = seg_pred.permute(0, 2, 1).contiguous().view(-1, self.args.num_segmentation_type)
                labels = labels.view(-1, 1)[:, 0].data  # array(batch_size*num_points)
                redundant_data_mask = redundant_data_mask.view(-1)

                redundant_data_mask = torch.where(redundant_data_mask == 1, True, False)
                seg_pred = seg_pred[redundant_data_mask, :]
                labels = labels[redundant_data_mask]

                loss = utilities.loss_calculation.loss_calculation(seg_pred,
                                                                   labels,
                                                                   using_weight=0)  # a scalar
                seg_pred = seg_pred.contiguous().view(-1, self.args.num_segmentation_type)
                pred_choice = seg_pred.data.max(1)[1]  # array(batch_size*num_points)

                points = einops.rearrange(raw_points, 'b n c -> (b n) c')
                points = points[redundant_data_mask, :]
                if all_predictions is None:
                    all_predictions = pred_choice
                    all_labels = labels
                    all_points = points

                else:
                    all_predictions = torch.cat((all_predictions, pred_choice), dim=0)
                    all_labels = torch.cat((all_labels, labels), dim=0)
                    all_points = torch.cat((all_points, points), dim=0)

                correct = torch.sum(pred_choice == labels)

                for i in range(self.args.num_segmentation_type):
                    for j in range(self.args.num_segmentation_type):
                        class_counts[i, j] = torch.sum((pred_choice == i) * (labels == j))

                total_correct += correct
                total_seen += (batch_size * self.args.npoints)
                loss_sum += loss

                for l in range(self.args.num_segmentation_type):
                    total_seen_class[l] += torch.sum((labels == l))
                    total_correct_class[l] += torch.sum((pred_choice == l) & (labels == l))
                    total_iou_deno_class[l] += torch.sum(((pred_choice == l) | (labels == l)))

            IoUs = (torch.Tensor(total_correct_class) / (torch.Tensor(total_iou_deno_class).float() + 1e-6))
            mIoU = IoUs.sum() / self.args.num_existing_type  # torch.mean(IoUs)

            test_class_acc = torch.sum(
                torch.tensor(total_correct_class) / (torch.tensor(total_seen_class).float() + 1e-6))
            test_class_acc = test_class_acc / self.args.num_existing_type

            test_loss = loss_sum / num_test_batch

            # total_correct = torch.tensor(total_correct)
            total_seen = torch.tensor(total_seen)
            test_point_acc = total_correct / float(total_seen)

            outstr = 'test loss %.6f, test point acc %.6f, test avg class acc %.6f' % (
                test_loss,
                test_point_acc,
                test_class_acc
            )
            print(outstr)
            print('test mean ioU %.6f' % mIoU)

            util.get_result_distribution_matrix(self.args.num_segmentation_type, all_predictions, all_labels,
                                                xyticks=list(label_rgb_dic.keys()),
                                                show_plt=False, plt_save_dir=self.plt_save_dir, sqrt_value=False)
            util.get_result_distribution_matrix(self.args.num_segmentation_type, all_predictions, all_labels,
                                                xyticks=list(label_rgb_dic.keys()),
                                                show_plt=False, plt_save_dir=self.plt_save_dir, sqrt_value=True)

            if self.save_dir is not None:
                save_pcd(all_points, all_predictions, self.save_dir)


def _pipeline_visualize_3_test_models():
    files_folder = r'E:\datasets\agiprobot\fromJan\pcd_from_raw_data_18\large_motor_tscan_npy'
    save_dir = r'E:\datasets\agiprobot\train_Output\2023_09_28_21_15' + r'\visualization'
    check_point_dir = r'E:\datasets\agiprobot\train_Output\2023_09_28_21_15\checkpoints\0.8212851881980896_best_finetune.pth'
    config_dir = 'D:\Jupyter\AgiProbot\large_motor_segmentation\config\segmentation_fine_tune_4_5_100epoch.yaml'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_name in os.listdir(files_folder):
        if '011' not in file_name:
            if '002' not in file_name:
                if '004' not in file_name:
                    continue

        data_set_dir = os.path.join(files_folder, file_name)
        pcd_save_dir = os.path.join(save_dir, file_name.split('.')[0] + '.pcd')
        plt_save_dir = os.path.join(save_dir, file_name.split('.')[0] + '.png')
        print(pcd_save_dir)
        tester = TestTrainedModel(
            config_dir=config_dir,
            model_name='pct pipeline',
            data_set_dir=data_set_dir,
            check_point_dir=check_point_dir,
            pcd_save_dir=pcd_save_dir,
            plt_save_dir=plt_save_dir
        )
        tester.run_test()


def _pipeline_test_trained_model():
    file_dir = '2023_09_30_21_29'
    pth_name = '0.8759808540344238_best_finetune.pth'
    config_dir = 'segmentation_fine_tune_4_5_500epoch_weights.yaml'

    files_folder = r'E:\datasets\agiprobot\fromJan\pcd_from_raw_data_18\large_motor_tscan_npy'
    save_dir = 'E:/datasets/agiprobot/train_Output/' + file_dir + r'/visualization'
    check_point_dir = 'E:/datasets/agiprobot/train_Output/' + file_dir + '/checkpoints/' + pth_name
    config_dir = 'D:/Jupyter/AgiProbot/large_motor_segmentation/config/' + config_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_name in os.listdir(files_folder):
        data_set_dir = os.path.join(files_folder, file_name)
        pcd_save_dir = os.path.join(save_dir, file_name.split('.')[0] + '.pcd')
        plt_save_dir = os.path.join(save_dir, file_name.split('.')[0] + '.png')
        print(pcd_save_dir)
        tester = TestTrainedModel(
            config_dir=config_dir,
            model_name='pct pipeline',
            data_set_dir=data_set_dir,
            check_point_dir=check_point_dir,
            pcd_save_dir=pcd_save_dir,
            plt_save_dir=plt_save_dir
        )
        tester.run_test()
    # output should be the same size as input


if __name__ == "__main__":
    # _pipeline_test_trained_model()
    # _pipeline_visualize_3_test_models()
    _pipeline_test_trained_model()
