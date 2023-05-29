import numpy as np
import open3d
import os
from tqdm import tqdm

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
                    points.append(np.array([x, y, z]))
                    point_color = rgb_dic[point_label]
                    colors.append(np.array([a / 255.0 for a in point_color]))

        self.points = np.array(points)
        self.colors = np.array(colors)

        print(self.points.shape)
        print(self.colors.shape)

        return self.points, self.colors

    def read_directory(self, dir, save_dir):
        for dir_name in os.listdir(dir):
            for i, file_name in enumerate(tqdm(os.listdir(dir + '/' + dir_name),
                                               total=len(os.listdir(dir + '/' + dir_name)), smoothing=0.9)):

                points = []
                colors = []
                labels = []
                with open(dir + '/' + dir_name + '/' + file_name, 'r') as f:

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

                        if x == '0' and y == '0' and z == '0':
                            continue

                        points.append(np.array([x, y, z]))
                        if label == '0':
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
                    save_name = 'Validation_' + file_name.split('.')[0] + '_' + dir_name
                else:
                    save_name = 'Training_' + file_name.split('.')[0] + '_' + dir_name

                np.save(os.path.join(save_dir, save_name), point_cloud)

    def save_and_visual_pcd(self, save_dir=None, visualization=False):
        '''
        save the file as *.pcd for further usage and visualization

        :param visualization:
        :param save_dir: direction to save this file
        :return:
        '''

        assert self.points is not None, 'read pcd file first!'

        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(self.points)
        point_cloud.colors = open3d.utility.Vector3dVector(self.colors)

        if save_dir is not None:
            open3d.io.write_point_cloud(save_dir, point_cloud, write_ascii=True)

        if visualization:
            open3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    pcd_reader = PcdReader()
    pcd_reader.read_directory('E:/datasets/agiprobot/binary_label/zivid', 'C:/Users/Lenovo/Desktop/numpy')
