import os
import open3d as o3d
import numpy as np


def check_color_one_pcd(colors, colors_list):
    for i in range(colors.shape[0]):
        colors_row = str(colors[i, :])
        if colors_row not in colors_list:
            colors_list.append(colors_row)
            print(colors_row)

    return colors_list


def check_pcd_label():
    labels = []
    files_direction_list = ['E:/datasets/agiprobot/agi_large_motor_dataset/02_05_22_n1',
                            'E:/datasets/agiprobot/agi_large_motor_dataset/02_05_22_n2',
                            'E:/datasets/agiprobot/agi_large_motor_dataset/03_05_22']

    for files_direction in files_direction_list:
        flag = True
        for subdir in os.listdir(files_direction):
            file_direction = files_direction + '/' + subdir
            for file_name in os.listdir(file_direction):
                if '0001.pcd' in file_name:

                    with open(file_direction + '/' + file_name, 'r') as f:

                        head_flag = True
                        while True:
                            # for i in range(12):
                            oneline = f.readline()

                            if head_flag:
                                if 'DATA ascii' in oneline:
                                    head_flag = False
                                    continue
                                else:
                                    continue

                            if not oneline:
                                break

                            x, y, z, _, label = list(oneline.strip('\n').split(' '))  # '0 0 0 1646617 8 -1\n'

                            if flag:
                                print(type(label))
                                flag = False
                            if label not in labels:
                                labels.append(label)
                                print(label)


def read_pcd(pcd_file):
    rgb_dic = {'808464432': [0, 0, 128],
               '892679477': [102, 255, 102],
               '909522486': [102, 255, 102],
               '858993459': [102, 255, 102],
               '825307441': [102, 255, 102],
               '875836468': [102, 255, 102],
               '842150450': [102, 255, 102]
               }
    points = []
    colors = []

    with open(pcd_file, 'r') as f:

        head_flag = True
        while True:
            # for i in range(12):
            oneline = f.readline()

            if head_flag:
                if 'DATA ascii' in oneline:
                    head_flag = False
                    continue
                else:
                    continue

            if not oneline:
                break

            x, y, z, _, label = list(oneline.strip('\n').split(' '))  # '0 0 0 1646617 8 -1\n'

            if label != '-1':
                points.append(np.array([x, y, z]))
                point_color = rgb_dic[label]
                colors.append(np.array([a / 255.0 for a in point_color]))

    points = np.array(points)
    colors = np.array(colors)
    print(points.shape)
    print(colors.shape)

    return points, colors


def pcd_to_binary1(files_direction, save_direction):
    '''
    pcd_to_binary1(files_direction, save_direction)
    relabel the form blendar generated pcd data, then save then under save_direction

    Args:
        files_direction (str): Path to pcd files.

    :return: None
    '''
    color_list = []
    black_ground_colors = ['[0.00392157 0.0745098  0.63921569]', '[0.8 0.8 0.8]']

    for subdir in os.listdir(files_direction):
        file_direction = files_direction + '/' + subdir
        for file_name in os.listdir(file_direction):
            if '0001.pcd' in file_name:
                pcd = o3d.io.read_point_cloud(file_direction + '/' + file_name, remove_nan_points=True,
                                              remove_infinite_points=True)

                colors_array = np.array(pcd.colors)
                print(np.array(pcd.label).shape)

                for i in range(colors_array.shape[0]):
                    color = str(colors_array[i, :])
                    if color in black_ground_colors:
                        colors_array[i, 0] = 0
                        colors_array[i, 1] = 0
                        colors_array[i, 2] = 128.0 / 255.0
                    else:
                        '''colors_array[i, 0] = 102.0 / 255.0
                        colors_array[i, 1] = 255.0 / 255.0
                        colors_array[i, 2] = 102.0 / 255.0'''
                        pass

                pcd.colors = o3d.utility.Vector3dVector(colors_array)
                o3d.io.write_point_cloud(save_direction + '/' + file_name, pcd, write_ascii=True)

                color_list = check_color_one_pcd(colors_array, color_list)


def pcd_to_binary2(files_direction, save_direction):
    '''
    pcd_to_binary2(files_direction, save_direction)
    relabel the form blendar generated pcd data,
    then save then under save_direction
        background rgb: [0, 0, 128],
        Motor rgb: [102, 255, 102]

    Args:
        files_direction (str): Path to pcd files.
        save_direction (str): Path to save.

    :return: None
    '''

    batch_name = files_direction.split('/')[-1]
    for subdir in os.listdir(files_direction):
        file_direction = files_direction + '/' + subdir
        for file_name in os.listdir(file_direction):
            if '0001.pcd' in file_name:
                points_array, colors_array = read_pcd(file_direction + '/' + file_name)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_array)
                pcd.colors = o3d.utility.Vector3dVector(colors_array)
                o3d.io.write_point_cloud(save_direction + '/' + batch_name + '_' + file_name, pcd, write_ascii=True)


def pcd_to_binary3(file_name, save_direction):
    '''
    pcd_to_binary3(pcd_file, save_direction)
    relabel the pcd data, which was scanned by zivid and labelled manually.
    then save then under save_direction
        background rgb: [0, 0, 128],
        Motor rgb: [102, 255, 102]

    Args:
        pcd_file (str): Path to pcd file.
        save_direction (str): Path to save.

    :return: None
    '''

    points_list = []
    colors_list = []

    with open(file_name, 'r') as f:

        head_flag = True
        while True:
            # for i in range(12):
            oneline = f.readline()

            if head_flag:
                if 'DATA ascii' in oneline:
                    head_flag = False
                    continue
                else:
                    continue

            if not oneline:
                break

            oneline = list(oneline.strip('\n').split(' '))

            if len(oneline) == 6:
                x, y, z, _, label, _ = list(oneline.strip('\n').split(' '))  # '0 0 0 1646617 8 -1\n'

                if label != '8':
                    if label == '1':
                        point_color = [0, 0, 128]
                    else:
                        point_color = [102, 255, 102]

                    points_list.append(np.array([x, y, z]))
                    colors_list.append(np.array([a / 255.0 for a in point_color]))

    points_array = np.array(points_list)
    colors_array = np.array(colors_list)
    print(points_list.shape)
    print(colors_list.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.colors = o3d.utility.Vector3dVector(colors_array)
    o3d.io.write_point_cloud(save_direction + '/' + file_name, pcd, write_ascii=True)


def pcd_to_binary4(file_name, save_name):
    '''
    pcd_to_binary4(pcd_file, save_name)
    relabel the pcd data, which was scanned by zivid and labelled manually.
    then save then under save_direction
        background rgb: [0, 0, 128],
        Motor rgb: [102, 255, 102]

    Args:
        pcd_file (str): Path to pcd file.
        save_name (str): Path to save.

    :return: None
    '''

    pcd = o3d.io.read_point_cloud(file_name, remove_nan_points=True, remove_infinite_points=True)

    colors_array = np.array(pcd.colors)

    for i in range(colors_array.shape[0]):
        color = list(colors_array[i, :])

        if color == [0., 0., 0.5019607843137255]:
            colors_array[i, 0] = 0
            colors_array[i, 1] = 0
            colors_array[i, 2] = 128.0 / 255.0
        else:
            colors_array[i, 0] = 102.0 / 255
            colors_array[i, 1] = 1.0
            colors_array[i, 2] = 102.0 / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors_array)
    o3d.io.write_point_cloud(save_name, pcd, write_ascii=True)


def get_motor_pcd(pcd_file, save_name):
    '''
    get_motor_pcd(pcd_file, save_name)
    get motor-only pcd file from relabeled pcd file
    then save then under save_name

    Args:
        pcd_file (str): Path to pcd file.
        save_name (str): Path to save.

    :return: None
    '''

    pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=True, remove_infinite_points=True)

    points_array = np.array(pcd.points)
    colors_array = np.array(pcd.colors)

    points_new_file = []
    colors_new_file = []

    print(colors_array.shape[0])

    for i in range(colors_array.shape[0]):

        if list(colors_array[i, :]) == [0., 0., 0.5019607843137255]:
            continue
        else:
            points_new_file.append(points_array[i, :])
            colors_new_file.append(colors_array[i, :])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_new_file)
    pcd.colors = o3d.utility.Vector3dVector(colors_new_file)
    o3d.io.write_point_cloud(save_name, pcd, write_ascii=True)

    print(np.array(colors_new_file).shape[0])


if __name__ == "__main__":
    '''files_direction_list = ['E:/datasets/agiprobot/agi_large_motor_dataset/02_05_22_n1',
                            'E:/datasets/agiprobot/agi_large_motor_dataset/02_05_22_n2',
                            'E:/datasets/agiprobot/agi_large_motor_dataset/03_05_22']
    for files_direction in files_direction_list:
        pcd_to_binary2(files_direction, 'C:/Users/Lenovo/Desktop/big_motor_blendar_binlabel')'''

    # check_pcd_label()

    '''files_direction_list = ['E:/datasets/agiprobot/agi_large_motor_dataset/02_05_22_n1',
                            'E:/datasets/agiprobot/agi_large_motor_dataset/02_05_22_n2',
                            'E:/datasets/agiprobot/agi_large_motor_dataset/03_05_22']
    for files_direction in files_direction_list:
        pcd_to_binary3(files_direction, 'C:/Users/Lenovo/Desktop/big_motor_blendar_binlabel')'''

    # pcd_to_binary4(r'E:\datasets\agiprobot\Video snippets\one_view.pcd', r'C:\Users\Lenovo\Desktop\one_view_bin.pcd')
    get_motor_pcd(r'E:\datasets\agiprobot\binlabel\one_view_bin.pcd',
                  r'E:\datasets\agiprobot\binlabel\one_view_motor_only.pcd')
    get_motor_pcd(r'E:\datasets\agiprobot\binlabel\big_motor_blendar_binlabel\02_05_22_n1_0scan_Motor_0001.pcd',
                  r'E:\datasets\agiprobot\binlabel\02_05_22_n1_0scan_Motor_0001_motor_only.pcd')

    pass
