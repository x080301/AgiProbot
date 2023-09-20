import torch
from utilities.config import get_parser
from torch.utils.data import DataLoader
from data_preprocess.data_loader import MotorDatasetTest
from tqdm import tqdm

from utilities import util
import models


def _pipeline_test_trained_model():
    tester = TestTrainedModel(
        config_dir='D:\Jupyter\AgiProbot\large_motor_segmentation\config\segmentation_fine_tune_5_7_100epoch.yaml',
        model_name='pct pipeline',
        data_set_dir=r'E:\datasets\agiprobot\fromJan\pcd_from_raw_data_18\large_motor_tscan_npy\Training_Motor_001_Motor_001.npy',
        check_point_dir=r'E:\datasets\agiprobot\train_Output\2023_09_18_02_26\checkpoints\0.779434084892273_best_finetune.pth',
        save_dir=r'C:\Users\Lenovo\Desktop\results'
    )
    tester.run_test()


class TestTrainedModel():
    def __init__(self, config_dir, model_name, data_set_dir, check_point_dir, save_dir=None):

        self.args = get_parser(config_dir=config_dir)

        if model_name == 'pct pipeline':
            from models.pct_token import PCTPipeline
            self.model = PCTPipeline(self.args)
        elif model_name == 'pct token':
            from models.pct_token import PCTToken
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

        test_dataset = MotorDatasetTest(data_set_dir, self.args.num_segmentation_type, self.args.npoints)
        self.test_data_loader = DataLoader(test_dataset,
                                           batch_size=self.args.test_batch_size,
                                           shuffle=True,
                                           drop_last=False)

        self.save_dir = save_dir

        self.criterion = util.cal_loss()

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

            for i, (points, seg) in tqdm(enumerate(self.test_data_loader),
                                         total=len(self.test_data_loader),
                                         smoothing=0.9):

                points, seg = points.cuda(non_blocking=True), seg.cuda(non_blocking=True)
                points = util.normalize_data(points)
                points, _ = util.rotate_per_batch(points, None)
                points = points.permute(0, 2, 1)
                batch_size = points.size()[0]

                seg_pred, trans = self.model(points.float())

                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                batch_label = seg.view(-1, 1)[:, 0].data  # array(batch_size*num_points)
                loss = util.cal_loss_pretrain(seg_pred.view(-1, self.args.num_segmentation_type),
                                              seg.view(-1, 1).squeeze(),
                                              using_weight=self.args.use_class_weight)  # a scalar
                seg_pred = seg_pred.contiguous().view(-1, self.args.num_segmentation_type)
                pred_choice = seg_pred.data.max(1)[1]  # array(batch_size*num_points)
                correct = torch.sum(pred_choice == batch_label)

                for i in range(self.args.num_segmentation_type):
                    for j in range(self.args.num_segmentation_type):
                        class_counts[i, j] = torch.sum((pred_choice == i) * (batch_label == j))

                total_correct += correct
                total_seen += (batch_size * self.args.npoints)
                loss_sum += loss

                for l in range(self.args.num_segmentation_type):
                    total_seen_class[l] += torch.sum((batch_label == l))
                    total_correct_class[l] += torch.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class[l] += torch.sum(((pred_choice == l) | (batch_label == l)))

            IoUs = (torch.Tensor(total_correct_class) / (torch.Tensor(total_iou_deno_class).float() + 1e-6))
            mIoU = IoUs.sum() / self.args.num_existing_type  # torch.mean(IoUs)
            torch.distributed.all_reduce(IoUs)
            torch.distributed.all_reduce(mIoU)

            test_class_acc = torch.sum(
                torch.tensor(total_correct_class) / (torch.tensor(total_seen_class).float() + 1e-6))
            test_class_acc = test_class_acc / self.args.num_existing_type
            torch.distributed.all_reduce(test_class_acc)

            torch.distributed.all_reduce(loss_sum)
            test_loss = loss_sum / num_test_batch

            # total_correct = torch.tensor(total_correct)
            total_seen = torch.tensor(total_seen)
            torch.distributed.all_reduce(total_correct)
            torch.distributed.all_reduce(total_seen)
            test_point_acc = total_correct / float(total_seen)

            outstr = 'test loss %.6f, test point acc %.6f, test avg class acc %.6f' % (
                test_loss,
                test_point_acc,
                test_class_acc
            )
            print(outstr)
            print('test mean ioU %.6f' % mIoU)

            util.get_result_distribution_matrix()


if __name__ == "__main__":
    _pipeline_test_trained_model()
