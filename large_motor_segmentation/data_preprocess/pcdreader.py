import numpy as np
import os
from tqdm import tqdm
import open3d as o3d
import torch

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


class PcdReader:
    points = None
    colors = None

    def read_pcd_ASCII(self, pcd_file_dir):
        '''
        read the output pcd file of the segmentation tool

        :param pcd_file_dir: direction of the .pcd file
        :return:
            points: N×3 numpy array, x,y,z position of the points
            colors: N×3 numpy array, normalized r,g,b of the points. The colour is chosen according to rgb_dic.
        '''

        points = []
        colors = []
        with open(pcd_file_dir, 'r') as f:

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

                x, y, z, _, label, _ = list(oneline.strip('\n').split(' '))  # '0 0 0 1646617 8 -1\n'

                point_label = list(rgb_dic.keys())[int(label)]
                if point_label != 'Noise':
                    points.append(np.array([-float(x), -float(y), float(z)]))
                    point_color = rgb_dic[point_label]
                    colors.append(np.array([a / 255.0 for a in point_color]))

        self.points = np.array(points)
        self.colors = np.array(colors)

        # print(self.points.shape)
        # print(self.colors.shape)

        return self.points, self.colors

    def read_directory(self, dir, save_dir):
        for i, root, _, file_name in enumerate(os.walk(dir)):
            # for dir_name in os.listdir(dir):
            #     for i, file_name in enumerate(tqdm(os.listdir(root + '/' + dir_name),
            #                                        total=len(os.listdir(root + '/' + dir_name)), smoothing=0.9)):

            points = []
            colors = []
            labels = []
            with open(os.path.join(root, file_name), 'r') as f:

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

                    x, y, z, label, _ = list(oneline.strip('\n').split(' '))  # '0 0 0 1646617 8 -1\n'

                    if x == '0' and y == '0' and z == '0':
                        continue
                    x, y, z, label = float(x), float(y), float(z), int(label)
                    points.append(np.array([x, y, z]))
                    if label == 0:
                        colors.append(np.array([0, 0, 0]))
                        labels.append(1)
                    else:
                        colors.append(np.array([0.5, 0.5, 0.5]))
                        labels.append(0)

            points = np.asarray(points)
            colors = np.asarray(colors)
            labels = np.asarray(labels)

            point_cloud = np.concatenate([points, colors], axis=-1)
            point_cloud = np.column_stack((point_cloud, labels))

            if i % 5 == 0:
                save_name = 'Validation_' + file_name.split('.')[0] + '_' + root.split('/')[-1]
            else:
                save_name = 'Training_' + file_name.split('.')[0] + '_' + root.split('/')[-1]

            np.save(os.path.join(save_dir, save_name), point_cloud)

    def save_and_visual_pcd(self, save_dir=None, visualization=False):
        '''
        save the file as *.pcd for further usage and visualization

        :param visualization:
        :param save_dir: direction to save this file
        :return:
        '''

        assert self.points is not None, 'read pcd file first!'

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points)
        point_cloud.colors = o3d.utility.Vector3dVector(self.colors)

        if save_dir is not None:
            o3d.io.write_point_cloud(save_dir, point_cloud, write_ascii=True)

        if visualization:
            o3d.visualization.draw_geometries([point_cloud])


def _pipeline_change_local_color():
    labels = []
    with open(r'C:\Users\Lenovo\Desktop\full_model.pcd', 'r') as f:
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

            x, y, z, _, label, _ = list(oneline.strip('\n').split(' '))  # '0 0 0 1646617 8 -1\n'

            point_label = list(rgb_dic.keys())[int(label)]
            if point_label != 'Noise':
                labels.append(int(label) == 1)

    labels = np.array(labels)

    source_pcd = o3d.io.read_point_cloud(r'E:\datasets\agiprobot\registration\full_model.pcd')

    colors = torch.asarray(np.asarray(source_pcd.colors))
    color = torch.asarray([a / 255.0 for a in rgb_dic['Main Housing']]).double()
    colors[labels, :] = color

    source_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename=r'E:\datasets\agiprobot\registration\full_model_2.pcd', pointcloud=source_pcd,
                             write_ascii=True)


def _pipeline_generate_background_only_pcd():
    source_pcd = o3d.io.read_point_cloud(r'E:\datasets\agiprobot\registration\one_view_bin.pcd',
                                         remove_nan_points=True,
                                         remove_infinite_points=True,
                                         print_progress=True)

    print(source_pcd)
    points = torch.asarray(np.asarray(source_pcd.points))
    colors = torch.asarray(np.asarray(source_pcd.colors))

    index = colors[:, 1]
    index = [a < 0.1 for a in index]

    background_only_pcd = o3d.geometry.PointCloud()
    background_only_pcd.points = o3d.utility.Vector3dVector(points[index, :])
    background_only_pcd.colors = o3d.utility.Vector3dVector(colors[index, :])
    print(background_only_pcd)

    o3d.io.write_point_cloud(filename=r'E:\datasets\agiprobot\registration\one_view_background_only.pcd',
                             pointcloud=background_only_pcd,
                             write_ascii=True)


def _pipeline_get_screw_only_pcd():
    source_pcd = o3d.io.read_point_cloud(r'E:\datasets\agiprobot\registration\full_model_2.pcd',
                                         remove_nan_points=True,
                                         remove_infinite_points=True,
                                         print_progress=True)

    print(source_pcd)
    points = torch.asarray(np.asarray(source_pcd.points))
    colors = torch.asarray(np.asarray(source_pcd.colors))

    index = colors[:, 2]
    index = [(a < 0.35) and (a > 0.3) for a in index]

    screw_only_pcd = o3d.geometry.PointCloud()
    screw_only_pcd.points = o3d.utility.Vector3dVector(points[index, :])
    screw_only_pcd.colors = o3d.utility.Vector3dVector(colors[index, :])
    print(screw_only_pcd)

    o3d.io.write_point_cloud(filename=r'E:\datasets\agiprobot\registration\screw_only_only.pcd',
                             pointcloud=screw_only_pcd,
                             write_ascii=True)


def _pipeline_color_pcd_from_label_tool():
    pcdreader = PcdReader()
    pcdreader.read_pcd_ASCII(r'C:\Users\Lenovo\Desktop\screw_only_only.pcd')
    pcdreader.save_and_visual_pcd(r'E:\datasets\agiprobot\registration\screw_only_only_colored.pcd')


def _pipeline_color_pcd_from_label_tool_directory():
    pcdreader = PcdReader()

    read_dir = r'E:\datasets\agiprobot\pipeline_demo\1'
    save_dir = r'E:\datasets\agiprobot\pipeline_demo'
    for file_name in tqdm(os.listdir(read_dir)):
        pcdreader.read_pcd_ASCII(os.path.join(read_dir, file_name))
        pcdreader.save_and_visual_pcd(os.path.join(save_dir, file_name))


if __name__ == "__main__":
    '''
    pcd_reader = PcdReader()
    pcd_reader.read_pcd_ASCII('E:/datasets/agiprobot/fromJan/pcd_tscan_31/labeled/_Motor_006_rot.pcd')
    pcd_reader.save_and_visual_pcd('E:/datasets/agiprobot/fromJan/pcd_tscan_31/labeled/_Motor_006_rot_1.pcd', visualization=True)
    '''
    # _pipeline_change_local_color()
    # _pipeline_generate_background_only_pcd()
    # _pipeline_get_screw_only_pcd()
    # _pipeline_color_pcd_from_label_tool()
    _pipeline_color_pcd_from_label_tool_directory()
