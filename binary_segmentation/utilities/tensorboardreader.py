from tensorboard.backend.event_processing import event_accumulator
import os
import matplotlib.pyplot as plt

keys = ['lr', 'loss/train_loss', 'point_acc/train_point_acc', 'mIoU/train_mIoU', 'IoU_background/train_IoU_background',
        'IoU_motor/train_IoU_motor', 'loss/eval_loss', 'point_acc/eval_point_acc', 'point_acc/eval_class_acc',
        'mIoU/eval_mIoU', 'IoU_background/eval_IoU_background', 'IoU_motor/eval_IoU_motor']
train_case_dic = {'2023_05_11_13_09_2048': '2048', '2023_05_10_06_00_4096': '4096',
                  '2023_05_11_02_29_4096_no_weights': '4096,no weights',
                  '2023_05_09_19_32_4096_sample_rate1': '4096,sample rate 1', '2023_05_11_23_00_1024': '1024',
                  '2023_05_12_14_48_4096points_400epoch': '4096, 400 epoches',
                  '2023_05_13_16_01_fine_tune': 'fine_tune'}

mIoU_dict = {}
for train_case in os.listdir('E:/datasets/agiprobot/train_Output'):
    if not train_case == '2023_05_05_21_24':
        directory = 'E:/datasets/agiprobot/train_Output/' + train_case + '/tensorboard_log'
        for file_name in os.listdir(directory):
            ea = event_accumulator.EventAccumulator(directory + '/' + file_name)
            ea.Reload()
            mIoU = ea.scalars.Items('mIoU/eval_mIoU')
            mIoU_x = [i.step for i in mIoU]
            mIoU_y = [i.value for i in mIoU]

            mIoU_dict[train_case_dic[train_case]] = [mIoU_x, mIoU_y]

show_list_0 = ['2048', '4096', '4096,no weights', '4096,sample rate 1', '1024', '4096, 400 epoches', 'fine_tune']
show_list = ['4096', '4096, 400 epoches', 'fine_tune']
plt.title("epoch and fine_tune")
for key in show_list:
    x = mIoU_dict[key][0]
    y = mIoU_dict[key][1]
    plt.plot(x, y, label=key)

# Set y-axis to logarithmic scale
plt.yscale('log')

# Set the y-axis limits
plt.ylim(0.95, 1)

# Add title and axis labels

plt.xlabel("epoch")
plt.ylabel("val mIoU (log scale)")

# Show the legend with data labels
plt.legend()

# Display the chart
plt.show()
