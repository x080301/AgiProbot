from tensorboard.backend.event_processing import event_accumulator
import os
import matplotlib.pyplot as plt
import numpy as np

keys = ['lr', 'loss/train_loss', 'point_acc/train_point_acc', 'mIoU/train_mIoU', 'IoU_background/train_IoU_background',
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
