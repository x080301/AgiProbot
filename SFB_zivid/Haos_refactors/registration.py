import time
import os
import open3d as o3d
import numpy as np
import copy
import sys


def point2point(target_point_cloud, source_point_cloud, max_correspondence_distance=10):
    pipreg = o3d.pipelines.registration

    reg = pipreg.registration_icp(source_point_cloud, target_point_cloud,
                                  max_correspondence_distance=max_correspondence_distance,
                                  estimation_method=
                                  # pipreg.TransformationEstimationPointToPlane())
                                  # pipreg.TransformationEstimationForGeneralizedICP())
                                  # pipreg.TransformationEstimationForColoredICP())
                                  pipreg.TransformationEstimationPointToPoint())

    print(reg.transformation)
    print(reg)
    print(type(reg.transformation))

    result_point_cloud = source_point_cloud  # copy.deepcopy(source_point_cloud)
    result_point_cloud = result_point_cloud.transform(reg.transformation)

    return result_point_cloud


def point2plane(target_point_cloud, source_point_cloud, max_correspondence_distance=10,
                evaluate_coarse_registraion_min_correspindence=None):
    radius = 1  # 5  # 1 # 0.5 # 0.1 # 0.01  # max search radius
    max_nn = 30  # max points in the search sphere
    source_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    pipreg = o3d.pipelines.registration
    # target and source are swapped, since target is full model and computation of its normal is easier.
    reg = pipreg.registration_icp(target_point_cloud, source_point_cloud,
                                  max_correspondence_distance=max_correspondence_distance,
                                  estimation_method=
                                  pipreg.TransformationEstimationPointToPlane()
                                  # pipreg.TransformationEstimationForGeneralizedICP()
                                  # pipreg.TransformationEstimationForColoredICP()
                                  # pipreg.TransformationEstimationPointToPoint()
                                  )

    print(reg)

    if evaluate_coarse_registraion_min_correspindence is not None \
            and np.array(reg.correspondence_set).shape[0] < evaluate_coarse_registraion_min_correspindence:

        return None

    else:

        result_point_cloud = source_point_cloud  # copy.deepcopy(source_point_cloud)
        # result_point_cloud = result_point_cloud.transform(transformation_matrix)

        # Transform
        # https://de.mathworks.com/matlabcentral/answers/321703-how-can-i-calculate-the-reverse-of-a-rotation-and-translation-that-maps-a-cloud-of-points-c1-to-an
        rotation_matrix = np.asarray(reg.transformation[0:3, 0:3])
        translation_vector = reg.transformation[0:3, 3]

        result_points = np.asarray(result_point_cloud.points)
        result_points = (result_points - translation_vector).reshape((-1, 3, 1))

        # print(np.linalg.inv(rotation_matrix).shape)
        # print(result_points.shape)
        result_points = (np.linalg.inv(rotation_matrix) @ result_points).reshape((-1, 3))

        result_point_cloud.points = o3d.utility.Vector3dVector(result_points)

        return result_point_cloud


class CoarseRegistrationExceptin(Exception):
    "this is user's Exception for unsuccessful coarse registration "

    def __init__(self):
        pass

    def __str__(self):
        print("coarse registration failed")


def zivid_3d_registration(target_point_cloud, source_point_cloud, rotation_per_capture,
                          algorithm='point2plane_multi_step',
                          visualization=False):
    '''

    :param target_point_cloud:
    :param source_point_cloud:
    :param algorithm: (string, optimal) 'point2plane_multi_step' or 'point2point_multi_step'
    :param visualization:
    :return: registered point cloud
    '''

    if visualization:
        time.perf_counter()
    # target_point_cloud = get_motor_only_pcd(target_point_cloud)

    if algorithm == 'point2plane_multi_step':
        fine_registration = point2plane
    elif algorithm == 'point2point_multi_step':
        fine_registration = point2point

    # global registration and first fine registration
    # coarse_registered = _coarse_registration_hard_coding(target_point_cloud, source_point_cloud)

    registered_point_cloud = copy.deepcopy(
        source_point_cloud)  # global_registration(target_point_cloud, copy.deepcopy(source_point_cloud))

    if rotation_per_capture != 0:
        euler_angle = registered_point_cloud.get_rotation_matrix_from_xyz((0, 0, np.pi * rotation_per_capture / 180))
        registered_point_cloud = registered_point_cloud.rotate(euler_angle)
        # o3d.visualization.draw_geometries([registered_point_cloud, target_point_cloud])

    if visualization:
        registered_point_cloud_v = copy.deepcopy(registered_point_cloud)
        registered_point_cloud_v.paint_uniform_color([1, 1, 0])
        target_point_cloud_v = copy.deepcopy(target_point_cloud)
        target_point_cloud_v.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([registered_point_cloud_v, target_point_cloud_v])

        # When the coarse registration give an unsuitable,
        # the correspindence set of the fine registration will be too small.
        # The size of correspindence set is checked here,
        # too small -> registration_algorithm() returns None

    registered_point_cloud = fine_registration(target_point_cloud, registered_point_cloud,
                                               max_correspondence_distance=5,
                                               evaluate_coarse_registraion_min_correspindence=10000)

    if registered_point_cloud is None:
        raise CoarseRegistrationExceptin

    # registered = registration_algorithm(target_point_cloud, registered, max_correspondence_distance=1)
    registered_point_cloud = fine_registration(target_point_cloud, registered_point_cloud,
                                               max_correspondence_distance=0.2)

    if visualization:
        print(time.perf_counter())
        registered_point_cloud_v = copy.deepcopy(registered_point_cloud)
        registered_point_cloud_v.paint_uniform_color([1, 1, 0])
        target_point_cloud_v = copy.deepcopy(target_point_cloud)
        target_point_cloud_v.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([registered_point_cloud_v, target_point_cloud_v])

    return registered_point_cloud


def registration_in_folders(registrted_folder=[],
                            pcd_directory='/home/wbk-ur2/dual_ws/src/agiprobot_control/scripts/SFB_Demo/models',
                            visualization=False):
    for root, _, files in os.walk(pcd_directory):

        if root in registrted_folder:
            continue
        else:
            registrted_folder.append(root)

        if len(files) == 0:
            continue

        combined_point_cloud = None

        files.sort()
        for file_name in files:
            if 'pc_part_' in file_name:
                file_direction = os.path.join(root, file_name)
                target_point_cloud = o3d.io.read_point_cloud(file_direction,
                                                             remove_nan_points=True,
                                                             remove_infinite_points=True,
                                                             print_progress=True)
                target_point_cloud, _ = target_point_cloud.remove_radius_outlier(nb_points=4, radius=0.5)
                target_point_cloud = target_point_cloud.voxel_down_sample(voxel_size=0.1)
                # o3d.visualization.draw_geometries([target_point_cloud])
                if combined_point_cloud is None:
                    combined_point_cloud = target_point_cloud
                else:
                    print(file_name)
                    if '8' in file_name:

                        combined_point_cloud = zivid_3d_registration(target_point_cloud,
                                                                     combined_point_cloud,
                                                                     -45) + target_point_cloud
                    else:
                        combined_point_cloud = zivid_3d_registration(target_point_cloud,
                                                                     combined_point_cloud,
                                                                     0) + target_point_cloud

                    combined_point_cloud = combined_point_cloud.voxel_down_sample(voxel_size=0.1)

        if visualization:
            o3d.visualization.draw_geometries([combined_point_cloud])

        o3d.io.write_point_cloud(filename=os.path.join(root, 'combined.pcd'),
                                 pointcloud=combined_point_cloud)


if __name__ == "__main__":
    '''
    target_point_cloud = o3d.io.read_point_cloud('/home/wbk-ur2/dual_ws/src/agiprobot_control/scripts/SFB_Demo/models/scan_5/pc_part_8.pcd',
                                                             remove_nan_points=True,
                                                              remove_infinite_points=True,
                                                            print_progress=True)
    source_point_cloud = o3d.io.read_point_cloud('/home/wbk-ur2/dual_ws/src/agiprobot_control/scripts/SFB_Demo/models/scan_5/pc_part_7.pcd',
                                                             remove_nan_points=True,
                                                              remove_infinite_points=True,
                                                             print_progress=True)
    
    rotation_per_capture=45
    euler_angle = target_point_cloud.get_rotation_matrix_from_xyz((0, 0, np.pi * rotation_per_capture / 180))
    target_point_cloud.rotate(euler_angle)

    o3d.visualization.draw_geometries([target_point_cloud,source_point_cloud])
    '''

    registration_in_folders(
        pcd_directory=r'/home/wbk-ur2/dual_ws/src/agiprobot_control/scripts/SFB_Demo/models/scan_32',
        visualization=True)
