import subprocess
import os
import shutil
import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
import scipy
from tqdm import tqdm

from utilities import point_cloud_operation


def run_cmd(cmd_str='', echo_print=False):
    """
    执行cmd命令，不显示执行过程中弹出的黑框
    备注：subprocess.run()函数会将本来打印到cmd上的内容打印到python执行界面上，所以避免了出现cmd弹出框的问题
    :param cmd_str: 执行的cmd命令
    :return:
    """
    if echo_print:
        print('\n执行cmd指令="{}"'.format(cmd_str))
    subprocess.run(cmd_str, shell=True)


def generate_pcd_without_label(source_dir, save_dir):
    for file_dir_batch in os.listdir(source_dir):
        for file_dir in os.listdir(source_dir + file_dir_batch):
            for file_name in os.listdir(source_dir + file_dir_batch + '/' + file_dir):
                if 'Szene.blend' not in file_name:
                    continue
                else:
                    # ******************* #
                    #   raw point cloud without label
                    # ******************* #
                    # source转存到intermediate results
                    shutil.copyfile(source_dir + file_dir_batch + '/' + file_dir + '/' + file_name,
                                    'D:/softwares/blender/Intermediate_results/blender_scene.blend')

                    # 转存到intermediate results
                    site = {"source_blender": "Intermediate_results/blender_scene.blend",
                            "py_script": "D:/Jupyter/AgiProbot/large_motor_segmentation/blender/pyscript_in_blender.py"}
                    cmd_str = 'D: & ' \
                              'cd /softwares/blender/ & ' \
                              'blender.exe --background {source_blender} --python {py_script}'.format(**site)
                    run_cmd(cmd_str=cmd_str, echo_print=False)

                    # 取点云，存到需要的位置
                    print('sampling')
                    point_cloud_operation.read_one_mesh(
                        'D:/softwares/blender/Intermediate_results/intermediate_obj.obj',

                        save_dir + file_dir_batch + '_' + file_dir + '.pcd',
                        number_of_points=200000
                    )


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


def set_points_colors(raw_points_pcd, all_parts_pcd):
    raw_points = np.asarray(raw_points_pcd.points)
    all_parts_pcd_points = np.asarray(all_parts_pcd.points)
    all_parts_pcd_colors = np.asarray(all_parts_pcd.colors)

    # set raw_points_pcd color according to the nearest point in all parts pcd
    # closest_point_index = cdist(raw_points, all_parts_pcd_points).argmin(axis=1)
    # raw_points_colors = all_parts_pcd_colors[closest_point_index]

    raw_points_colors = np.empty((0, 3))
    for i in tqdm(range(0, raw_points.shape[0], 1000)):
        closest_point_index = cdist(raw_points[i:i + 1000, :], all_parts_pcd_points).argmin(axis=1)

        raw_points_colors = np.concatenate((raw_points_colors, all_parts_pcd_colors[closest_point_index, :]), axis=0)

    raw_points_pcd.colors = o3d.utility.Vector3dVector(raw_points_colors)

    return raw_points_pcd


def generate_pcd_with_label(source_dir, save_dir):
    for file_dir_batch in os.listdir(source_dir):
        for file_dir in os.listdir(source_dir + file_dir_batch):
            for file_name in os.listdir(source_dir + file_dir_batch + '/' + file_dir):
                if 'Szene.blend' not in file_name:
                    continue
                else:

                    # ******************* #
                    # raw point cloud without label
                    # ******************* #
                    # source转存到intermediate results
                    shutil.copyfile(source_dir + file_dir_batch + '/' + file_dir + '/' + file_name,
                                    'D:/softwares/blender/Intermediate_results/blender_scene.blend')

                    # 转存到intermediate results
                    site = {"source_blender": "Intermediate_results/blender_scene.blend",
                            "py_script": "D:/Jupyter/AgiProbot/large_motor_segmentation/blender/pyscript_in_blender.py"}
                    cmd_str = 'D: & ' \
                              'cd /softwares/blender/ & ' \
                              'blender.exe --background {source_blender} --python {py_script}'.format(**site)
                    run_cmd(cmd_str=cmd_str, echo_print=False)

                    # obj转点云
                    print('sampling')
                    raw_points_pcd = point_cloud_operation.read_one_mesh(
                        'D:/softwares/blender/Intermediate_results/intermediate_obj.obj',
                        number_of_points=200000
                    )
                    # np.save('D:/softwares/blender/Intermediate_results/points.npy', np.asarray(raw_points_pcd.points))
                    # raw_points_pcd = o3d.geometry.PointCloud()
                    # raw_points_pcd.points = o3d.utility.Vector3dVector(
                    #     np.load('D:/softwares/blender/Intermediate_results/points.npy'))

                    # raw_points = np.asarray(raw_points_pcd.points)

                    # ******************* #
                    # points in parts with label
                    # ******************* #
                    # 保存各部分的obj文件
                    site = {
                        "source_blender": "Intermediate_results/blender_scene.blend",
                        "py_script": "D:/Jupyter/AgiProbot/large_motor_segmentation/blender/generate_obj_with_label_using_blender.py"}
                    cmd_str = 'D: & ' \
                              'cd /softwares/blender/ & ' \
                              'blender.exe --background {source_blender} --python {py_script}'.format(**site)
                    run_cmd(cmd_str=cmd_str, echo_print=False)

                    # 读取obj并转pcd
                    all_parts_pcd = None
                    for part_obj_file_name in os.listdir('D:/softwares/blender/Intermediate_results/'):
                        if '_part.obj' in part_obj_file_name:
                            part_pcd = point_cloud_operation.read_one_mesh(
                                'D:/softwares/blender/Intermediate_results/' + part_obj_file_name,
                                number_of_points=50000)

                            rgb_color = [a / 255.0 for a in rgb_dic[part_obj_file_name.split('_')[1]]]
                            part_pcd.paint_uniform_color(rgb_color)

                            if all_parts_pcd is None:
                                all_parts_pcd = part_pcd
                            else:
                                all_parts_pcd += part_pcd

                    # 将各部分的颜色投影到无颜色的pcd中
                    pcd_with_color = set_points_colors(raw_points_pcd, all_parts_pcd)

                    o3d.io.write_point_cloud(save_dir + file_dir_batch + '_' + file_dir + '.pcd', pcd_with_color)
                    # o3d.visualization.draw_geometries([pcd_with_color])

                    # part_points = np.asarray(part_pcd.points)
                    # print(part_points.shape)


if __name__ == "__main__":
    generate_pcd_with_label('E:/datasets/agiprobot/agi_large_motor_dataset_syn/',
                            'E:/datasets/agiprobot/large_motor_syn/labeled_pcd/')
