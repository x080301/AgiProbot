import numpy as np
import open3d as o3d
import os


def resave_binary_labelled_pcd(read_directory, save_directory):
    for file_name in os.listdir(read_directory):
        point_cloud = o3d.io.read_point_cloud(os.path.join(read_directory, file_name), remove_nan_points=True,
                                              remove_infinite_points=True, print_progress=True)

        colors = np.asarray(point_cloud.colors)
        points = np.asarray(point_cloud.points)
        point_cloud = np.concatenate([points, colors], axis=-1)

        point_cloud[:, 3] *= 225
        point_cloud[:, 4] *= 225
        point_cloud[:, 5] *= 225

        label = np.sum(point_cloud[:, 3:6], axis=1)
        label = np.where(label > 150, 1, 0)  # 0:bacground, 1: motor

        point_cloud = np.column_stack((point_cloud, label))

        np.save(os.path.join(save_directory, file_name), point_cloud)


if __name__ == "__main__":
    resave_binary_labelled_pcd('E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel',
                               'E:/datasets/agiprobot/binary_label/big_motor_blendar_binlabel_npy')
