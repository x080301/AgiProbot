import subprocess
import os
import shutil

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


blender_file_dir = 'E:/datasets/agiprobot/agi_large_motor_dataset_syn/'
for file_dir_batch in os.listdir(blender_file_dir):
    for file_dir in os.listdir(blender_file_dir + file_dir_batch):
        for file_name in os.listdir(blender_file_dir + file_dir_batch + '/' + file_dir):
            if 'Szene.blend' not in file_name:
                continue
            else:
                # source转存到intermediate results
                shutil.copyfile(blender_file_dir + file_dir_batch + '/' + file_dir + '/' + file_name,
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
                    'E:/datasets/agiprobot/large_motor_syn/raw_pcd/' + file_dir_batch + '_' + file_dir + '.pcd',
                    number_of_points=200000
                )

r'''
mesh = o3d.io.read_triangle_mesh(r'C:\Users\Lenovo\Desktop\0Szene.obj')
# pcd = mesh.sample_points_uniformly(number_of_points=10000)
# print(pcd.points)
o3d.visualization.draw_geometries([mesh], window_name=".obj")

'''
