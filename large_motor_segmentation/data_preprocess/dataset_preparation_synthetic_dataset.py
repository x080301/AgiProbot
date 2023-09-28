import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

rgb_dic = {'Void': [207, 207, 207],
           'Background': [0, 0, 128],
           'Gear': [102, 140, 255],
           'Connector': [102, 255, 102],
           'Screws': [247, 77, 77],
           'Solenoid': [255, 165, 0],
           'Electrical Connector': [255, 255, 0],
           'Main Housing': [0, 100, 0],
           'Noise': [223, 200, 200],
           'Inner Gear': [107, 218, 250]
           }


def resave_labelled_pcd_as_npy(read_directory, save_directory, debug_visualization=False, label_rgb_dic=None,
                               valid_list=None, excluding_list=[]):
    """
    read *.pcd, choose the segment label according to its color, and resave it as *.npy
    :param label_rgb_dic:
    :param read_directory: from where, *.pcd is read.
    :param save_directory: to where, *.npy is saved
    :param debug_visualization:
    :return: no return
    """
    if label_rgb_dic is None:
        label_rgb_dic = rgb_dic

    rgb_dic_values = list(label_rgb_dic.values())
    rgb_dic_values_sum = [sum(x) for x in rgb_dic_values]  # [621, 128, 497, 459, 401, 420, 510, 100, 623, 575]

    i = 0
    for root, _, files in os.walk(read_directory):
        for file_name in tqdm(files):
            if file_name in excluding_list:
                continue

            point_cloud = o3d.io.read_point_cloud(os.path.join(root, file_name), remove_nan_points=True,
                                                  remove_infinite_points=True, print_progress=True)

            colors = np.asarray(point_cloud.colors)
            points = np.asarray(point_cloud.points)

            point_cloud = np.concatenate([points, colors], axis=-1)

            label = []

            rgb_sum = colors.sum(axis=1)
            rgb_sum = np.around(rgb_sum * 255)
            for j in range(points.shape[0]):
                rgb_sum_j = int(rgb_sum[j])
                label.append(rgb_dic_values_sum.index(rgb_sum_j))

            if debug_visualization:
                plt.figure(clear=True)

                plt.hist(label, bins=50, color='blue', alpha=0.7)
                plt.show()

            label = np.asarray(label)

            point_cloud = np.column_stack((point_cloud, label))

            if valid_list is None:
                if i % 5 == 0:
                    save_name = 'Validation_' + file_name.split('.')[0]
                else:
                    save_name = 'Training_' + file_name.split('.')[0]
                i += 1
            else:
                for test_name in valid_list:
                    if test_name in file_name:
                        save_name = 'Validation_' + file_name.split('.')[0]
                        break
                else:
                    save_name = 'Training_' + file_name.split('.')[0]

            np.save(os.path.join(save_directory, save_name), point_cloud)

            if debug_visualization:
                print(point_cloud[-1, :])
                unique_values, counts = np.unique(point_cloud[:, 6], return_counts=True)
                total_rows = point_cloud.shape[0]
                percentages = counts / total_rows

                for value, percentage in zip(unique_values, percentages):
                    print(percentage)


def prepare_token_label(save_path):
    """
    prepare token labels for every point cloud
    :param save_path:
    :return:
    """
    pass


def _pipline_resave_fine_tune_label():
    label_rgb_dic = {
        'Gear': [102, 140, 255],
        'Connector': [102, 255, 102],
        'Screws': [247, 77, 77],
        'Solenoid': [255, 165, 0],
        'Electrical Connector': [255, 255, 0],
        'Main Housing': [0, 100, 0]
    }
    valid_list = ['002', '004', '011']
    resave_labelled_pcd_as_npy(r'E:\datasets\agiprobot\fromJan\pcd_from_raw_data_18\colored',
                               r'E:\datasets\agiprobot\fromJan\pcd_from_raw_data_18\large_motor_tscan_npy',
                               label_rgb_dic=label_rgb_dic,
                               valid_list=valid_list,
                               excluding_list=['Motor_025_Motor_025.pcd']
                               # , debug_visualization=True
                               )


if __name__ == "__main__":
    _pipline_resave_fine_tune_label()
