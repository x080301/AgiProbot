import open3d as o3d
import numpy as np
from tqdm import tqdm
import os


def check_color_one_pcd(point_cloud_input_file_name):
    point_cloud = o3d.io.read_point_cloud(point_cloud_input_file_name,
                                          remove_nan_points=True, remove_infinite_points=True, print_progress=True)
    colors = np.asarray(point_cloud.colors)
    points = np.asarray(point_cloud.points)
    point_cloud = np.concatenate([points, colors], axis=-1)

    colors_list = []

    for i in range(point_cloud.shape[0]):
        colors_row = str(colors[i, :])
        if colors_row not in colors_list:
            colors_list.append(colors_row)
            print(colors_row)


def check_color_all(point_cloud_input_file_dir):
    colors_list = []
    for directory in os.listdir(point_cloud_input_file_dir):
        for sub_directory in tqdm(os.listdir(point_cloud_input_file_dir + '/' + directory)):
            for file_name in os.listdir(point_cloud_input_file_dir + '/' + directory + '/' + sub_directory):
                if 'Motor_0001.pcd' in file_name:
                    point_cloud_input_file_name = point_cloud_input_file_dir + '/' + directory + '/' + sub_directory + '/' + file_name
                    point_cloud = o3d.io.read_point_cloud(point_cloud_input_file_name,
                                                          remove_nan_points=True, remove_infinite_points=True,
                                                          print_progress=True)
                    colors = np.asarray(point_cloud.colors)
                    for i in range(colors.shape[0]):
                        colors_row = str(colors[i, :])
                        if colors_row not in colors_list:
                            colors_list.append(colors_row)
                            print(colors_row)
                            print(point_cloud_input_file_name)




if __name__ == "__main__":
    # check_color_one_pcd('E:/datasets/agiprobot/agi_large_motor_dataset/03_05_22/0/0scan_Motor_0001.pcd')
    # check_color_all('E:/datasets/agiprobot/agi_large_motor_dataset')

    ply_2_pcd(r'C:\Users\Lenovo\Desktop\untitled.ply',
              r'C:\Users\Lenovo\Desktop\MotorClean.plz.pcd')

    # point_cloud = o3d.io.read_point_cloud('E:/datasets/agiprobot/TScan Schulung/MotorClean.plz.pcd',
    #                                       remove_nan_points=True, remove_infinite_points=True,
    #                                       print_progress=True)
    # colors = np.asarray(point_cloud.colors)
    # print(colors.shape)
    # print(np.unique(colors, axis=0))

    pass
