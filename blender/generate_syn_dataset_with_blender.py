import subprocess
import os
import shutil
import open3d as o3d
import numpy as np
import scipy
from tqdm import tqdm
import torch

# This color assignment is unified in project agiprobot.
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


def read_one_mesh(file_dir=r"C:\Users\Lenovo\Desktop\Alignment\Alignment\Motor_001.stl", save_dir=None,
                  number_of_points=200000):
    """
    :param save_dir:
    :param number_of_points:
    :param file_dir: direction of *.stl file
    :return: point cloud
    """

    mesh = o3d.io.read_triangle_mesh(file_dir)

    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)  # 600000
    if save_dir is not None:
        pcd.paint_uniform_color([1, 1, 0])
        o3d.io.write_point_cloud(save_dir, pcd, write_ascii=True)

    return pcd


def set_points_colors_torch(raw_points_pcd, all_parts_pcd):
    raw_points = torch.tensor(np.asarray(raw_points_pcd.points)).cuda(non_blocking=True)
    all_parts_pcd_points = torch.tensor(np.asarray(all_parts_pcd.points)).cuda(non_blocking=True)
    all_parts_pcd_colors = torch.tensor(np.asarray(all_parts_pcd.colors)).cuda(non_blocking=True)

    # set raw_points_pcd color according to the nearest point in all parts pcd

    raw_points_colors = torch.empty((0, 3))
    for i in tqdm(range(0, raw_points.shape[0], 1000)):
        get_manhattan_distance = torch.nn.PairwiseDistance(p=1)
        manhattan_distance = get_manhattan_distance(raw_points[i:i + 1000, :], all_parts_pcd_points)
        _, closest_point_index = manhattan_distance.topk(k=1, dim=-1)

        raw_points_colors = np.concatenate((raw_points_colors, all_parts_pcd_colors[closest_point_index, :]), axis=0)

    raw_points_pcd.colors = o3d.utility.Vector3dVector(raw_points_colors)

    return raw_points_pcd


def set_points_colors(raw_points_pcd, all_parts_pcd):
    raw_points = np.asarray(raw_points_pcd.points)
    all_parts_pcd_points = np.asarray(all_parts_pcd.points)
    all_parts_pcd_colors = np.asarray(all_parts_pcd.colors)

    # set raw_points_pcd color according to the nearest point in all parts pcd

    raw_points_colors = np.empty((0, 3))
    for i in tqdm(range(0, raw_points.shape[0], 1000)):
        closest_point_index = scipy.spatial.distance.cdist(raw_points[i:i + 1000, :], all_parts_pcd_points).argmin(
            axis=1)

        raw_points_colors = np.concatenate((raw_points_colors, all_parts_pcd_colors[closest_point_index, :]), axis=0)

    raw_points_pcd.colors = o3d.utility.Vector3dVector(raw_points_colors)

    return raw_points_pcd


def generate_pcd_with_label(source_dir, save_dir):
    """
    generate pcd files from blender models
    !!!update cmd_str before use!!! /softwares/blender/ should be redirected to your own blender.exe direction!

    :param source_dir: directory where blender models are located as .blend files
    :param save_dir: directory where to save the results as .pcd files
    :return: no return
    """
    generated_pcd_list = os.listdir(save_dir)

    for root, _, files in os.walk(source_dir):
        for file_name in files:
            if 'Szene.blend' not in file_name:
                continue

            save_path = root.split('/')[-1].split('\\')[0] + '_' + root.split('/')[-1].split('\\')[1] + '.pcd'
            if save_path in generated_pcd_list:
                continue

            # ******************* #
            # raw point cloud without label
            # ******************* #
            # copy source to intermediate results
            shutil.copyfile(os.path.join(root, file_name),
                            'D:/softwares/blender/Intermediate_results/blender_scene.blend')

            # run python script in blender with Windows CMD to generate an .obj file,
            # which contains the synthetic large motor model without label.
            # python script: pyscript_in_blender.py
            site = {"source_blender": "Intermediate_results/blender_scene.blend",
                    "py_script": os.path.dirname(__file__) + "/blender_scripts/" +
                                 "pyscript_in_blender.py"}
            cmd_str = 'D: & ' \
                      'cd /softwares/blender/ & ' \
                      'blender.exe --background {source_blender} --python {py_script}'.format(**site)
            subprocess.run(cmd_str, shell=True)

            # load obj and sample it to point cloud
            print('sampling')
            whole_model_pcd = read_one_mesh(
                'D:/softwares/blender/Intermediate_results/intermediate_obj.obj',
                number_of_points=600000
            )

            # ******************* #
            # points in parts with label
            # ******************* #
            # run python script in blender to generate a series of .obj files.
            # Each .obj file contains objects of one semantic type.
            # python script: generate_obj_with_label_using_blender.py
            site = {
                "source_blender": "Intermediate_results/blender_scene.blend",
                "py_script": os.path.dirname(__file__) + '/blender_scripts/'
                             + 'generate_obj_with_label_using_blender.py'
            }
            cmd_str = 'D: & ' \
                      'cd /softwares/blender/ & ' \
                      'blender.exe --background {source_blender} --python {py_script}'.format(**site)
            subprocess.run(cmd_str, shell=True)

            # load obj and sample it to point cloud
            # set color according to its semantic type
            part_pcd = None
            for part_obj_file_name in os.listdir('D:/softwares/blender/Intermediate_results/'):
                if '_part.obj' in part_obj_file_name:
                    part_pcd_one = read_one_mesh(
                        'D:/softwares/blender/Intermediate_results/' + part_obj_file_name,
                        number_of_points=50000)

                    rgb_color = [a / 255.0 for a in rgb_dic[part_obj_file_name.split('_')[1]]]
                    part_pcd_one.paint_uniform_color(rgb_color)

                    if part_pcd is None:
                        part_pcd = part_pcd_one
                    else:
                        part_pcd += part_pcd_one
            # o3d.visualization.draw_geometries([part_pcd])

            # whole_model_pcd: point cloud sampled from the synthetic large motor model without label
            # part_pcd: point cloud sampled from files.
            #           Each of them contains objects of one semantic type.
            # set color of points in whole_model_pcd.
            # The color should be the same as the closest point to itself in part_pcd.
            # pcd_with_color = set_points_colors_cuda(whole_model_pcd, part_pcd)
            pcd_with_color = set_points_colors(whole_model_pcd, part_pcd)

            # save the result: pcd_with_color
            o3d.io.write_point_cloud(
                os.path.join(
                    save_dir,
                    save_path),
                pcd_with_color)


if __name__ == "__main__":
    generate_pcd_with_label('E:/datasets/agiprobot/large_motor_syn/raw_blender_model/',
                            'E:/datasets/agiprobot/large_motor_syn/labeled_pcd_600million_points/')
