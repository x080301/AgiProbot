import open3d as o3d

from registration import registration


def _pipeline_registration_and_save():
    target_point_cloud = o3d.io.read_point_cloud(r'E:\datasets\agiprobot\registration\one_view_motor_only.pcd',
                                                 remove_nan_points=True,
                                                 remove_infinite_points=True,
                                                 print_progress=True)
    source_point_cloud = o3d.io.read_point_cloud(r'E:\datasets\agiprobot\registration\full_model_2.pcd',
                                                 remove_nan_points=True,
                                                 remove_infinite_points=True,
                                                 print_progress=True)

    registered_point_cloud = registration(target_point_cloud, source_point_cloud)

    o3d.io.write_point_cloud(filename=r'E:\datasets\agiprobot\registration\full_model_2_registered.pcd',
                             pointcloud=registered_point_cloud,
                             write_ascii=True)
if __name__ == "__main__":
    _pipeline_registration_and_save()