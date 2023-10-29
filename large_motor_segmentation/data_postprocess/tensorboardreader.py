from tensorboard.backend.event_processing import event_accumulator
import os
import matplotlib.pyplot as plt
import numpy as np


def _alt_pipeline():
    keys = ['lr', 'loss/train_loss', 'point_acc/train_point_acc', 'mIoU/train_mIoU',
            'IoU_background/train_IoU_background',
            'IoU_motor/train_IoU_motor', 'loss/eval_loss', 'point_acc/eval_point_acc', 'point_acc/eval_class_acc',
            'mIoU/eval_mIoU', 'IoU_background/eval_IoU_background', 'IoU_motor/eval_IoU_motor']
    train_case_dic = {'2023_05_11_13_09_2048': '2048', '2023_05_10_06_00_4096': '4096',
                      '2023_05_11_02_29_4096_no_weights': '4096,no weights',
                      '2023_05_09_19_32_4096_sample_rate1': '4096,sample rate 1', '2023_05_11_23_00_1024': '1024',
                      '2023_05_12_14_48_4096points_400epoch': '4096, 400 epoches',
                      '2023_05_13_16_01_fine_tune': 'fine_tune',
                      '2023_05_14_16_07_fine_tune2': 'fine_tune2',
                      '2023_05_17_07_15_old_lr': '2048, old_lr',
                      '2023_05_17_17_37_4096': '4096_no_weights',
                      '2023_05_18_14_15_2048_no_weights': '2048_no_weights',
                      '2023_05_18_03_53_withweights': '2048_with_weights',
                      '2023_05_19_02_11_4096_with_weights': '4096_with_weights',
                      '2023_05_28_03_29_without_stnloss': 'without_stnloss',
                      '2023_05_28_14_24_with_stnloss': 'with_stnloss',
                      '2023_05_30_02_39_pretrain': 'pre_trained',
                      '2023_05_30_15_27_zivid_5_8': 'zivid_5_8',
                      '2023_05_31_20_21_zivid_4_7': 'zivid_4_7',
                      '2023_05_31_10_07_lr1e4_6': 'lr 1e-4->-6',
                      '2023_06_03_04_59_lr_5_7': 'lr 1e-5->-7',
                      '2023_06_02_00_48_lr5_6': 'lr 1e-5->-6',
                      '2023_06_01_15_40_lr1e4_7': 'lr 1e-4->-7',
                      '2023_06_02_09_56_zivid_freeze': 'zivid_5_8_freeze',
                      '2023_07_12_21_49_4_7_no_pretrain': 'zivid_4_7_no_pretrain',
                      '2023_07_15_20_02_no_rotation_augument': 'lr 1e-4->-6 no_rotatoion_augument',
                      '2023_07_16_04_46_pretrain_1e51e7': 'pretrain 1e-5->1e-7',
                      '2023_07_16_13_33_fine_tune_5e71e8': 'fine tune 5e-7->1e-8',
                      '2023_07_17_08_19_pretain_5_6': '100 epoch pretrain 1e-5->1e-6',
                      '2023_07_17_17_06_fine_tune_6_8': 'fine tune 5e->6->1e-8',
                      '2023_07_18_11_54_100_epoch_pretain_4_6': '100 pretrain 1e-4->1e-6',
                      '2023_07_18_16_18_100_epoch_fine_tune_5_7': '100 fine tune 46_1e-5->1e-7',
                      '2023_07_20_05_52_100_epoch_pretain_4_5': '100 pretrain 1e-4->1e-5',
                      '2023_07_20_10_16_100_epoch_fine_tune_5_7': '100 fine tune 45_1e-5->1e-7',
                      '2023_07_25_00_46_full_model_pretrain_4_5_100epoch': 'full model pretrain 1e-4 -> 1e-5 100epoch',
                      '2023_07_26_00_10_full_model_pretrain_4_6_200epoch': 'full model pretrain 1e-4 -> 1e-6 200epoch',
                      '2023_10_02_09_08_56_np': '5->6 no pretrain',
                      '2023_10_02_06_37_56': '5->6',
                      '2023_10_01_21_14_46_np': '4->6 no pretrain',
                      '2023_10_01_19_20_46': '4->6 500epoch',
                      '2023_10_01_08_02_45_np': '4->5 no pretrain',
                      '2023_09_30_20_51_45': '4->5',
                      '2023_10_02_23_56_46_100ep': '4->6 100epoch',
                      '2023_10_03_02_26_46_200ep': '4->6 200epoch',
                      '2023_10_03_07_19_46_300ep': '4->6 300epoch',
                      '2023_10_03_14_55_weights': 'with weights 1',
                      '2023_10_03_19_29_weights2': 'with weights 1/2',
                      '2023_10_04_00_04_weights3': 'with weights 1/3'
                      }
    show_list = ['4->6 200epoch', 'with weights 1', 'with weights 1/2', 'with weights 1/3']
    plt.title("mIoU weights")

    mIoU_dict = {}
    for train_case in os.listdir('E:/datasets/agiprobot/train_Output'):
        if train_case in train_case_dic:
            directory = 'E:/datasets/agiprobot/train_Output/' + train_case + '/tensorboard_log'
            for file_name in os.listdir(directory):
                ea = event_accumulator.EventAccumulator(directory + '/' + file_name)
                ea.Reload()
                mIoU = ea.scalars.Items('mIoU/eval_mIoU')
                mIoU_x = [i.step for i in mIoU]
                mIoU_y = [i.value for i in mIoU]

                x_smooth = np.linspace(min(mIoU_x), max(mIoU_x), 30)
                y_smooth = np.interp(x_smooth, mIoU_x, mIoU_y)

                mIoU_dict[train_case_dic[train_case]] = [mIoU_x, mIoU_y, x_smooth, y_smooth]
    y_max = 0
    for key in show_list:
        x = mIoU_dict[key][0]
        y = mIoU_dict[key][1]  # [i / 6 * 8 for i in mIoU_dict[key][1]]
        plt.plot(x, y, color=(0.9, 0.9, 0.9))
        y_max = max(max(y), y_max)

    for i, key in enumerate(show_list):
        x_smooth = mIoU_dict[key][2]
        y_smooth = mIoU_dict[key][3]  # [i / 6 * 8 for i in mIoU_dict[key][3]]
        plt.plot(x_smooth, y_smooth, label=key)

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Set the y-axis limits
    plt.ylim(0.8, 0.91)
    xmin = 90
    xmax = 310
    plt.xlim(xmin, xmax)

    # Draw a horizontal line at the maximum y value
    plt.axhline(y_max, color='green', linestyle='--', label='max mIoU')
    plt.annotate(f'{y_max:.6f}', xy=(xmin + 5, y_max), xytext=(xmin, y_max + 5),
                 textcoords='offset points', ha='center', va='bottom')
    # Add title and axis labels

    plt.xlabel("epoch")
    plt.ylabel("val mIoU (log scale)")

    # Show the legend with data labels
    plt.legend()

    # Display the chart
    plt.show()


def _pipeline_valid_sample_for_cross_validation():
    train_case_dic = {
        '2023_10_03_02_26_46_200ep': '2,4',
        'cross_validation/2023_10_04_22_41_02&07': '2,7',
        'cross_validation/2023_10_05_03_13_02&08': '2,8',
        'cross_validation/2023_10_05_07_45_02&09': '2,9',
        'cross_validation/2023_10_05_12_17_02&17': '2,17',
        'cross_validation/2023_10_05_16_51_04&07': '4,7',
        'cross_validation/2023_10_05_21_24_04&08': '4,8',
        'cross_validation/2023_10_06_01_59_04&09': '4,9',
        'cross_validation/2023_10_06_03_38_04&17': '4,17',
        'cross_validation/2023_10_06_06_31_07&08': '7,8',
        'cross_validation/2023_10_06_08_08_07&09': '7,9',
        'cross_validation/2023_10_06_11_04_07&17': '7,17',
        'cross_validation/2023_10_06_12_33_08&09': '8,9',
        'cross_validation/2023_10_06_15_32_08&17': '8,17',
        'cross_validation/2023_10_06_16_56_09&17': '9,17'
    }
    show_list = ['2,4', '2,7', '2,8', '2,9', '2,17', '4,7', '4,8', '4,9', '4,17', '7,8', '7,9', '7,17', '8,9', '8,17',
                 '9,17']
    plt.title("mIoU weights")
    emphasize_list = ['9,17', '4,17', '2,4', '2,9', '9,17', '2,17']

    mIoU_dict = {}

    for train_case_direction in train_case_dic:
        directory = 'E:/datasets/agiprobot/train_Output/' + train_case_direction + '/tensorboard_log'
        for file_name in os.listdir(directory):
            ea = event_accumulator.EventAccumulator(directory + '/' + file_name)
            ea.Reload()
            mIoU = ea.scalars.Items('mIoU/eval_mIoU')
            mIoU_x = [i.step for i in mIoU]
            mIoU_y = [i.value for i in mIoU]

            x_smooth = np.linspace(min(mIoU_x), max(mIoU_x), 30)
            y_smooth = np.interp(x_smooth, mIoU_x, mIoU_y)

            mIoU_dict[train_case_dic[train_case_direction]] = [mIoU_x, mIoU_y, x_smooth, y_smooth]

    for key in show_list:
        x = mIoU_dict[key][0]
        y = mIoU_dict[key][1]  # [i / 6 * 8 for i in mIoU_dict[key][1]]
        #     plt.plot(x, y, color=(0.9, 0.9, 0.9))
        print(key + ': %(ymax)f' % {'ymax': max(y)})

    y_max = 0
    y_mean = 0
    for key in emphasize_list:
        x = mIoU_dict[key][0]
        y = mIoU_dict[key][1]  # [i / 6 * 8 for i in mIoU_dict[key][1]]
        #     plt.plot(x, y, color=(0.9, 0.9, 0.9))
        y_max = max(max(y), y_max)
        y_mean += max(y)
    y_mean = y_mean / len(emphasize_list)
    print(y_mean)

    for i, key in enumerate(show_list):

        x_smooth = mIoU_dict[key][2]
        y_smooth = mIoU_dict[key][3]  # [i / 6 * 8 for i in mIoU_dict[key][3]]
        if key in emphasize_list:
            plt.plot(x_smooth, y_smooth, label=key)
        else:
            plt.plot(x_smooth, y_smooth, color=(0.9, 0.9, 0.9))

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Set the y-axis limits
    plt.ylim(0.8, 0.91)
    xmin = 90
    xmax = 310
    plt.xlim(xmin, xmax)

    # Draw a horizontal line at the maximum y value
    plt.axhline(y_max, color='green', linestyle='--', label='max mIoU')
    plt.annotate(f'{y_max:.6f}', xy=(xmin + 5, y_max), xytext=(xmin, y_max + 5),
                 textcoords='offset points', ha='center', va='bottom')

    plt.axhline(y_mean, color='green', linestyle=':', label='mean mIoU')
    plt.annotate(f'{y_mean:.6f}', xy=(xmin + 5, y_mean), xytext=(xmin, y_mean + 5),
                 textcoords='offset points', ha='center', va='bottom')
    # Add title and axis labels

    plt.xlabel("epoch")
    plt.ylabel("val mIoU (log scale)")

    # Show the legend with data labels
    plt.legend()

    # Display the chart
    plt.show()


def _pipeline_valid_sample_for_cross_validation():
    train_case_dic = {
        '2023_10_03_02_26_46_200ep': '2,4',
        'cross_validation/2023_10_04_22_41_02&07': '2,7',
        'cross_validation/2023_10_05_03_13_02&08': '2,8',
        'cross_validation/2023_10_05_07_45_02&09': '2,9',
        'cross_validation/2023_10_05_12_17_02&17': '2,17',
        'cross_validation/2023_10_05_16_51_04&07': '4,7',
        'cross_validation/2023_10_05_21_24_04&08': '4,8',
        'cross_validation/2023_10_06_01_59_04&09': '4,9',
        'cross_validation/2023_10_06_03_38_04&17': '4,17',
        'cross_validation/2023_10_06_06_31_07&08': '7,8',
        'cross_validation/2023_10_06_08_08_07&09': '7,9',
        'cross_validation/2023_10_06_11_04_07&17': '7,17',
        'cross_validation/2023_10_06_12_33_08&09': '8,9',
        'cross_validation/2023_10_06_15_32_08&17': '8,17',
        'cross_validation/2023_10_06_16_56_09&17': '9,17',
        'cross_validation2/2023_10_11_17_14_08&05': '8,5',
        'cross_validation2/2023_10_10_10_10_03&13': '3,13',
        'cross_validation2/2023_10_10_05_15_03&06': '3,6',
        'cross_validation2/2023_10_10_00_50_03&05': '3,5',
        'cross_validation2/2023_10_10_15_06_03&18': '3,18',
        'cross_validation2/2023_10_11_22_11_09&03': '9,3',
        'cross_validation2/2023_10_11_14_48_03&04': '3,4',
        'cross_validation2/2023_10_11_19_43_09&06': '9,6',
        'cross_validation2/2023_10_11_12_08_03&36': '3,36'
    }
    show_list = ['8,5', '3,13', '3,6', '3,5', '3,18', '9,3', '3,4', '9,6',
                 '3,36',
                 '2,4', '2,7', '2,8', '2,9', '2,17', '4,7', '4,8', '4,9', '4,17', '7,8', '7,9', '7,17', '8,9', '8,17',
                 '9,17']
    plt.title("cross validation")
    # emphasize_list = ['8,5', '3,13', '3,6', '3,5', '3,18', '9,3', '3,4', '9,6', '3,36']
    # emphasize_list = ['9,17', '4,17', '2,4', '2,9', '9,17', '2,17']
    emphasize_list = ['8,17', '2,4', '4,17', '9,6', '3,13']

    mIoU_dict = {}

    for train_case_direction in train_case_dic:
        directory = 'E:/datasets/agiprobot/train_Output/' + train_case_direction + '/tensorboard_log'
        for file_name in os.listdir(directory):
            ea = event_accumulator.EventAccumulator(directory + '/' + file_name)
            ea.Reload()
            mIoU = ea.scalars.Items('mIoU/eval_mIoU')
            if '09&06' in train_case_direction:
                mIoU_x = [2 * (i.step - 97) + 97 for i in mIoU]
            else:
                mIoU_x = [i.step for i in mIoU]
            mIoU_y = [i.value for i in mIoU]

            x_smooth = np.linspace(min(mIoU_x), max(mIoU_x), 30)
            y_smooth = np.interp(x_smooth, mIoU_x, mIoU_y)

            mIoU_dict[train_case_dic[train_case_direction]] = [mIoU_x, mIoU_y, x_smooth, y_smooth]

    for key in show_list:
        x = mIoU_dict[key][0]
        y = mIoU_dict[key][1]  # [i / 6 * 8 for i in mIoU_dict[key][1]]
        #     plt.plot(x, y, color=(0.9, 0.9, 0.9))
        print(key + ': %(ymax)f' % {'ymax': max(y)})

    y_max = 0
    y_mean = 0
    for key in emphasize_list:
        x = mIoU_dict[key][0]
        y = mIoU_dict[key][1]  # [i / 6 * 8 for i in mIoU_dict[key][1]]
        #     plt.plot(x, y, color=(0.9, 0.9, 0.9))
        y_max = max(max(y), y_max)
        y_mean += max(y)
    y_mean = y_mean / len(emphasize_list)
    print(y_mean)

    for i, key in enumerate(show_list):

        x_smooth = mIoU_dict[key][2]
        y_smooth = mIoU_dict[key][3]  # [i / 6 * 8 for i in mIoU_dict[key][3]]
        if key in emphasize_list:
            plt.plot(x_smooth, y_smooth, label=key)
        else:
            plt.plot(x_smooth, y_smooth, color=(0.9, 0.9, 0.9))

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Set the y-axis limits
    plt.ylim(0.6, 0.91)
    xmin = 90
    xmax = 310
    plt.xlim(xmin, xmax)

    # Draw a horizontal line at the maximum y value
    plt.axhline(y_max, color='green', linestyle='--', label='max mIoU')
    plt.annotate(f'{y_max:.6f}', xy=(xmin + 5, y_max), xytext=(xmin, y_max + 5),
                 textcoords='offset points', ha='center', va='bottom')

    plt.axhline(y_mean, color='green', linestyle=':', label='mean mIoU')
    plt.annotate(f'{y_mean:.6f}', xy=(xmin + 5, y_mean), xytext=(xmin, y_mean + 5),
                 textcoords='offset points', ha='center', va='bottom')
    # Add title and axis labels

    plt.xlabel("epoch")
    plt.ylabel("val mIoU (log scale)")

    # Show the legend with data labels
    plt.legend()

    # Display the chart
    plt.show()


def _pipeline_valid_sample_visualization(train_case_dir=r'E:\datasets\agiprobot\train_Output\2023_10_15_10_43_pointnet',
                                         show='mIoU/eval_mIoU'):
    print(show)
    # train_case_dir = r'E:\datasets\agiprobot\train_Output\2023_10_15_10_43_pointnet'
    plt.title("DGCNN")

    y_max = 0
    y_mean = 0

    direcotry_list = os.listdir(train_case_dir)
    if 'checkpoints' in direcotry_list:
        direcotry_list = ['']

    for train_case_direction in direcotry_list:
        directory = train_case_dir + '/' + train_case_direction + '/tensorboard_log'
        # if '03-13' in directory:
        #     continue

        for file_name in os.listdir(directory):
            ea = event_accumulator.EventAccumulator(directory + '/' + file_name)
            ea.Reload()

            print(ea.scalars.Keys())
            mIoU = ea.scalars.Items(show)  # ('valid_IoU/Bolt')#('class_acc/eval_class_acc')#('mIoU/eval_mIoU')

        mIoU_x = [(i.step - 97) * 2 + 97 for i in mIoU]
        # mIoU_x = [i.step for i in mIoU if i.step<=197]
        mIoU_y = [i.value for i in mIoU]

        x_smooth = np.linspace(min(mIoU_x), max(mIoU_x), 30)
        y_smooth = np.interp(x_smooth, mIoU_x, mIoU_y)

        key = train_case_direction.split('_')[-1]

        plt.plot(mIoU_x, mIoU_y, color=(0.9, 0.9, 0.9))
        plt.plot(x_smooth, y_smooth, label=key)

        print(max(mIoU_y))
        y_max = max(max(mIoU_y), y_max)
        y_mean += max(mIoU_y)

    y_mean = y_mean / len(direcotry_list)
    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Set the y-axis limits
    # plt.ylim(0.5, 0.91)
    xmin = 90  # 90
    xmax = 310  # 310
    plt.xlim(xmin, xmax)

    # Draw a horizontal line at the maximum y value
    plt.axhline(y_max, color='green', linestyle='--', label='max mIoU')
    plt.annotate(f'{y_max:.6f}', xy=(xmin + 5, y_max), xytext=(xmin, y_max + 5),
                 textcoords='offset points', ha='center', va='bottom')

    plt.axhline(y_mean, color='green', linestyle=':', label='mean mIoU')
    plt.annotate(f'{y_mean:.6f}', xy=(xmin + 5, y_mean), xytext=(xmin, y_mean + 5),
                 textcoords='offset points', ha='center', va='bottom')
    # Add title and axis labels

    plt.xlabel("epoch")
    plt.ylabel("val mIoU (log scale)")

    # Show the legend with data labels
    plt.legend()

    # Display the chart
    plt.show()


def _mIoU_oa():
    train_case_dir = r'E:\datasets\agiprobot\train_Output\2023_07_20_05_52_100_epoch_pretain_4_5'


    _pipeline_valid_sample_visualization(train_case_dir=train_case_dir, show='point_acc/eval_point_acc')
    _pipeline_valid_sample_visualization(train_case_dir=train_case_dir, show='mIoU/eval_mIoU')
    # _pipeline_valid_sample_visualization(train_case_dir=train_case_dir, show='class_acc/eval_class_acc')
    # _pipeline_valid_sample_visualization(train_case_dir=train_case_dir, show='mIoU/eval_mIoU')
    # _pipeline_valid_sample_visualization(train_case_dir=train_case_dir, show='valid_IoU/Bolt')


def _double_train_log_epoch():
    from torch.utils.tensorboard import SummaryWriter

    dirs = r''

    for dir_name in os.listdir(dirs):
        train_log_dir = dirs + '\\' + dir_name + r'\tensorboard_log'

        for file_name in os.listdir(train_log_dir):
            ea = event_accumulator.EventAccumulator(train_log_dir + '/' + file_name)
            ea.Reload()

            os.remove(train_log_dir + '/' + file_name)

        log_writer = SummaryWriter(train_log_dir)

        for i in range(len(ea.scalars.Items('lr'))):

            for key in ea.scalars.Keys():
                # epoch = (ea.scalars.Items(key)[i].step - 97) * 2 + 97
                epoch = ea.scalars.Items(key)[i].step * 2
                log_writer.add_scalar(key, ea.scalars.Items(key)[i].value, epoch)


if __name__ == '__main__':
    _mIoU_oa()
