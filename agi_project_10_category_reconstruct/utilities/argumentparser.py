import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from torch.utils.tensorboard import SummaryWriter


def init_file_path(args, add_string, exp_name):
    io = None
    io_test = None
    # initial the file, if not exiting (os.path.exists() is pointed at ralative position and current cwd)
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/' + args.model + '/' + args.which_dataset):
        os.makedirs('outputs/' + args.model + '/' + args.which_dataset)
    if not os.path.exists(
            'outputs/' + args.model + '/' + args.which_dataset + '/' + exp_name + add_string + '/' + 'models'):
        os.makedirs('outputs/' + args.model + '/' + args.which_dataset + '/' + exp_name + add_string + '/' + 'models')
    if not args.test:
        if not os.path.exists(str(
                BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/0'):
            os.makedirs(str(
                BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/0')
            path = str(
                BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/0'
            writer = SummaryWriter(str(
                BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/0')
            io = PrintLog(str(
                BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/0' + '/run' + add_string + '.log')
        else:
            i = 1
            while (True):
                if not os.path.exists(str(
                        BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string
                                      + '/' + str(i)):
                    os.makedirs(str(
                        BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string
                                + '/' + str(
                        i))
                    path = str(
                        BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/' + str(
                        i)
                    writer = SummaryWriter(str(
                        BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string
                                           + '/' + str(i))
                    io = PrintLog(str(
                        BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string
                                  + '/' + str(i) + '/run' + add_string + '.log')
                    break
                else:
                    i += 1
    else:
        i = 0
        path = str(
            BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/' + str(
            i)
        while os.path.exists(str(
                BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/' + str(
            i)):
            path = str(
                BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/' + str(
                i)
            io_test = PrintLog(str(
                BASE_DIR) + "/outputs/" + args.model + '/' + args.which_dataset + '/' + args.exp_name + add_string + '/' + str(
                i) + '/Test' + add_string + '.log')
            i += 1
    return path, writer, io, io_test


class PrintLog():
    def __init__(self, path):
        self.f = open(path, 'a')  # 'a' is used to add some contents at end  of current file

    def c_print(self, text):
        print(text)
        text = str(text)
        self.f.write(text + '\n')
        self.f.flush()  # to ensure the line will be wroten and the content in buffer will get deleted

    def close(self):
        self.f.close()


def argumentparser():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--model', type=str, default='PCT', metavar='N',
                        choices=['PCT', 'PCT_patch'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--batch_size', type=int, default=3, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--data_dir', type=str, default='/home/bi/study/thesis/data_process/synthetic/merge',
                        help='file need to be tested')
    parser.add_argument('--which_dataset', type=str, default='test', metavar='N',
                        help='experiment version to record reslut')
    parser.add_argument('--exp_name', type=str, default='STN_16_2048_100_steptogether_10cate', metavar='N',
                        help='experiment version to record reslut')
    parser.add_argument('--training', type=int, default=1,
                        help='evaluate the model')
    parser.add_argument('--test', type=int, default=0,
                        help='evaluate the model')
    parser.add_argument('--finetune', type=int, default=0, metavar='N',
                        help='if we finetune the model')
    parser.add_argument('--num_segmentation_type', type=int, default=10, metavar='num_segmentation_type',
                        help='num_segmentation_type)')
    parser.add_argument('--kernel_loss_weight', type=float, default=0.05, metavar='F',
                        help='factor of loss_cluster')
    parser.add_argument('--stn_loss_weight', type=float, default=0.01, metavar='F',
                        help='factor of loss_cluster')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_class_weight', type=float, default=0,
                        help='enables using class weights(0 represents not using \
                            the class weight and 1 represent using the class weights')
    parser.add_argument('--screw_weight', type=float, default=1.0, metavar='F',
                        help='weight for screw')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--npoints', type=int, default=2048,
                        help='Point Number [default: 4096]')
    parser.add_argument('--validation_symbol', type=str, default='Validation',
                        help='Which datablocks to use for validation')
    parser.add_argument('--test_symbol', type=str, default='Test',
                        help='Which datablocks to use for test')
    parser.add_argument('--num_heads', type=int, default=4, metavar='num_attention_heads',
                        help='number of attention_heads for self_attention ')
    parser.add_argument('--num_attention_layer', type=int, default=1, metavar='num_attention_hea',
                        help='number of attention_heads for self_attention ')
    parser.add_argument('--self_encoder_latent_features', type=int, default=128, metavar='self_encoder_latent_features',
                        help='number of hidden size for self_attention ')
    parser.add_argument('--hidden_size_for_cross_attention', type=int, default=512,
                        metavar='hidden_size_for_cross_attention',
                        help='number of hidden size for cross attention')
    parser.add_argument('--after_stn_as_input', type=int, default=1, metavar='S',
                        help='get the neighbors from rotated points for each superpoints')
    parser.add_argument('--after_stn_as_kernel_neighbor_query', type=int, default=1, metavar='S',
                        help='get the high dimensional features for patch sample to get super points with rotated points')
    args = parser.parse_args()

    if args.finetune:
        add_string = '_finetune'
    else:
        add_string = ''

    path, writer, io, io_test = init_file_path(add_string, args.exp_name)

    return args, path, writer, io, io_test


if __name__ == '__main__':
    pass
