import open3d as o3d
import os
from tqdm import tqdm
import numpy as np


def crop_cuboid(point_cloud_dir, save_dir, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None,
                split=False):
    if '.pcd' in point_cloud_dir:
        pcd = o3d.io.read_point_cloud(point_cloud_dir)

        points = np.asarray(pcd.points)
        # colors = np.asarray(pcd.colors) * 255
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        bounding_box_list = [x_min, x_max, y_min, y_max, z_min, z_max]

        selection_bool = True
        is_min = True
        for i, axis_value in enumerate(bounding_box_list):
            if axis_value is not None:
                if is_min:
                    selection_bool = selection_bool & (points[:, i // 2] > axis_value)
                else:
                    selection_bool = selection_bool & (points[:, i // 2] < axis_value)
            # print(np.sum(selection_bool))

            is_min = not is_min

        selection_index, = np.where(selection_bool)

        pcd_outlier = pcd.select_by_index(selection_index, invert=True)

        outlier_save_file_name = os.path.join(save_dir, 'outlier_' + os.path.basename(point_cloud_dir))

        o3d.io.write_point_cloud(filename=outlier_save_file_name, pointcloud=pcd_outlier)

        if split:
            selection_index_1, = np.where(selection_bool & (points[:, 2] <= z_min + 50))
            selection_index_2, = np.where(selection_bool & (points[:, 2] > z_min + 50) & (points[:, 2] <= z_min + 100))
            selection_index_3, = np.where(selection_bool & (points[:, 2] > z_min + 100) & (points[:, 2] <= z_min + 150))
            selection_index_4, = np.where(selection_bool & (points[:, 2] > z_min + 150))
            pcd_inlier_part = pcd.select_by_index(selection_index_1, invert=False)
            save_file_name = os.path.join(save_dir,
                                          'inlier_' + os.path.basename(point_cloud_dir).split('.')[0] + '_1.pcd')
            o3d.io.write_point_cloud(filename=save_file_name, pointcloud=pcd_inlier_part)  # , write_ascii=True)

            pcd_inlier_part = pcd.select_by_index(selection_index_2, invert=False)
            save_file_name = os.path.join(save_dir,
                                          'inlier_' + os.path.basename(point_cloud_dir).split('.')[0] + '_2.pcd')
            o3d.io.write_point_cloud(filename=save_file_name, pointcloud=pcd_inlier_part)  # , write_ascii=True)

            pcd_inlier_part = pcd.select_by_index(selection_index_3, invert=False)
            save_file_name = os.path.join(save_dir,
                                          'inlier_' + os.path.basename(point_cloud_dir).split('.')[0] + '_3.pcd')
            o3d.io.write_point_cloud(filename=save_file_name, pointcloud=pcd_inlier_part)  # , write_ascii=True)

            pcd_inlier_part = pcd.select_by_index(selection_index_4, invert=False)
            save_file_name = os.path.join(save_dir,
                                          'inlier_' + os.path.basename(point_cloud_dir).split('.')[0] + '_4.pcd')
            o3d.io.write_point_cloud(filename=save_file_name, pointcloud=pcd_inlier_part)  # , write_ascii=True)
        else:
            pcd_inlier = pcd.select_by_index(selection_index, invert=False)
            inlier_save_file_name = os.path.join(save_dir, 'inlier_' + os.path.basename(point_cloud_dir))
            o3d.io.write_point_cloud(filename=inlier_save_file_name, pointcloud=pcd_inlier)

    else:

        for root, _, files in os.walk(point_cloud_dir):
            print('try reading pcds in: \n' + root + '\n')
            for file_name in tqdm(files):
                crop_cuboid(os.path.join(root, file_name), save_dir, x_min, x_max, y_min, y_max, z_min, z_max, split)


def _pipeline_crop_cuboid():
    crop_cuboid('E:/datasets/agiprobot/SFB_Demo/models/registered',
                'E:/datasets/agiprobot/SFB_Demo/models/cropped'
                , y_min=540
                , y_max=700
                , z_min=135
                , split=True
                )


if __name__ == "__main__":
    _pipeline_crop_cuboid()
