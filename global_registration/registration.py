import numpy as np
import open3d as o3d
import copy
import time

from data_visualization import visualization_point_cloud
from global_registration import global_registration


def _point2plane_test(target_point_cloud, source_point_cloud, max_correspondence_distance=10):
    pipreg = o3d.pipelines.registration

    reg = pipreg.registration_icp(source_point_cloud, target_point_cloud,
                                  max_correspondence_distance=max_correspondence_distance,
                                  estimation_method=
                                  pipreg.TransformationEstimationPointToPlane()
                                  # pipreg.TransformationEstimationForGeneralizedICP()
                                  # pipreg.TransformationEstimationForColoredICP()
                                  # pipreg.TransformationEstimationPointToPoint()
                                  )

    print(reg.transformation)
    print(reg)
    print(type(reg.transformation))

    result_point_cloud = copy.deepcopy(source_point_cloud)
    result_point_cloud = result_point_cloud.transform(reg.transformation)

    return result_point_cloud


def _translation(target_point_cloud, source_point_cloud):
    translation_vector = target_point_cloud.get_center() - source_point_cloud.get_center()

    source_point_cloud.translate(translation_vector, relative=True)

    return source_point_cloud


def _rotation(source_point_cloud):
    rotated_point_cloud = copy.deepcopy(source_point_cloud)

    euler_angle = rotated_point_cloud.get_rotation_matrix_from_xyz((-np.pi * 30. / 180., np.pi * 3 / 4., 0))
    rotated_point_cloud.rotate(euler_angle)

    return rotated_point_cloud


def _coarse_registration_hard_coding(target_point_cloud, source_point_cloud):
    translated_piont_cloud = _translation(target_point_cloud, source_point_cloud)
    rotated_point_cloud = _rotation(translated_piont_cloud)

    result_piont_cloud = rotated_point_cloud

    return result_piont_cloud


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


def get_motor_only_pcd(target_point_cloud):
    colors = np.array(target_point_cloud.colors)
    points = np.array(target_point_cloud.points)

    target_index = np.argwhere(np.sum(colors, axis=1) > 1)
    motor_only = points[target_index, :].reshape((-1, 3))

    motor_only_point_cloud = o3d.geometry.PointCloud()
    motor_only_point_cloud.points = o3d.utility.Vector3dVector(motor_only)

    return motor_only_point_cloud


class CoarseRegistrationExceptin(Exception):
    "this is user's Exception for unsuccessful coarse registration "

    def __init__(self):
        pass

    def __str__(self):
        print("coarse registration failed")


def registration(target_point_cloud, source_point_cloud, algorithm='point2plane_multi_step', visualization=False):
    '''

    :param target_point_cloud:
    :param source_point_cloud:
    :param algorithm: (string, optimal) 'point2plane_multi_step' or 'point2point_multi_step'
    :param visualization:
    :return: registered point cloud
    '''

    if visualization:
        time.perf_counter()
    target_point_cloud = get_motor_only_pcd(target_point_cloud)

    if algorithm == 'point2plane_multi_step':
        fine_registration = point2plane
    elif algorithm == 'point2point_multi_step':
        fine_registration = point2point

    # global registration and first fine registration
    for i in range(5):
        # coarse_registered = _coarse_registration_hard_coding(target_point_cloud, source_point_cloud)
        registered_point_cloud = global_registration(target_point_cloud, copy.deepcopy(source_point_cloud))

        registered_point_cloud.paint_uniform_color([1, 1, 0])

        if visualization:
            print("coarse_registration")
            visualization_point_cloud(source_point_cloud=registered_point_cloud,
                                      target_point_cloud_with_background=target_point_cloud)

        # When the coarse registration give an unsuitable,
        # the correspindence set of the fine registration will be too small.
        # The size of correspindence set is checked here,
        # too small -> registration_algorithm() returns None

        registered_point_cloud = fine_registration(target_point_cloud, registered_point_cloud,
                                                   max_correspondence_distance=5,
                                                   evaluate_coarse_registraion_min_correspindence=100000)

        if registered_point_cloud is not None:
            break
    else:
        raise CoarseRegistrationExceptin

    # registered = registration_algorithm(target_point_cloud, registered, max_correspondence_distance=1)
    registered_point_cloud = fine_registration(target_point_cloud, registered_point_cloud,
                                               max_correspondence_distance=0.2)

    if visualization:
        print(time.perf_counter())
        print("fine_registration")
        visualization_point_cloud(source_point_cloud=registered_point_cloud,
                                  target_point_cloud_with_background=target_point_cloud)

    return registered_point_cloud


def zivid_3d_registration(target_point_cloud, source_point_cloud, algorithm='point2plane_multi_step',
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
    euler_angle = registered_point_cloud.get_rotation_matrix_from_xyz((0, 0, np.pi * 45. / 180))
    registered_point_cloud.rotate(euler_angle)

    if visualization:
        registered_point_cloud_V = copy.deepcopy(
            registered_point_cloud)
        registered_point_cloud_V.paint_uniform_color([1, 1, 0])
        print("coarse_registration")
        visualization_point_cloud(source_point_cloud=registered_point_cloud_V,
                                  target_point_cloud_with_background=target_point_cloud)

    # When the coarse registration give an unsuitable,
    # the correspindence set of the fine registration will be too small.
    # The size of correspindence set is checked here,
    # too small -> registration_algorithm() returns None

    registered_point_cloud = fine_registration(target_point_cloud, registered_point_cloud,
                                               max_correspondence_distance=5,
                                               evaluate_coarse_registraion_min_correspindence=100000)

    if registered_point_cloud is None:
        raise CoarseRegistrationExceptin

    # registered = registration_algorithm(target_point_cloud, registered, max_correspondence_distance=1)
    registered_point_cloud = fine_registration(target_point_cloud, registered_point_cloud,
                                               max_correspondence_distance=0.2)

    if visualization:
        print(time.perf_counter())
        print("fine_registration")
        visualization_point_cloud(source_point_cloud=registered_point_cloud,
                                  target_point_cloud_with_background=target_point_cloud)

    return registered_point_cloud


if __name__ == "__main__":
    # data_generation()

    target_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_7.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)

    source_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_8.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)

    # rotated_point_cloud = copy.deepcopy(source_point_cloud)
    # '''-np.pi * 30. / 180'''
    # euler_angle = rotated_point_cloud.get_rotation_matrix_from_xyz((0, 0, -np.pi * 45. / 180))
    # rotated_point_cloud.rotate(euler_angle)

    # visualization_point_cloud(source_point_cloud=rotated_point_cloud,
    #                           target_point_cloud_with_background=target_point_cloud)
    registered_pcd = zivid_3d_registration(target_point_cloud, source_point_cloud)
    registered_pcd = registered_pcd + target_point_cloud

    target_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_6.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)
    registered_pcd = zivid_3d_registration(target_point_cloud, registered_pcd)
    registered_pcd = registered_pcd + target_point_cloud

    target_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_5.pcd',
                                                 remove_nan_points=True, remove_infinite_points=False,
                                                 print_progress=True)
    registered_pcd = zivid_3d_registration(target_point_cloud, registered_pcd)
    registered_pcd = registered_pcd + target_point_cloud

    o3d.visualization.draw_geometries([registered_pcd])

    target_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_4.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)
    registered_pcd = zivid_3d_registration(target_point_cloud, registered_pcd, visualization=True)
    registered_pcd = registered_pcd + target_point_cloud

    o3d.visualization.draw_geometries([registered_pcd])

    target_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_3.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)
    registered_pcd = zivid_3d_registration(target_point_cloud, registered_pcd, visualization=True)
    registered_pcd = registered_pcd + target_point_cloud

    o3d.visualization.draw_geometries([registered_pcd])

    target_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_2.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)
    registered_pcd = zivid_3d_registration(target_point_cloud, registered_pcd, visualization=True)
    registered_pcd = registered_pcd + target_point_cloud

    o3d.visualization.draw_geometries([registered_pcd])

    target_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_1.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)
    registered_pcd = zivid_3d_registration(target_point_cloud, registered_pcd, visualization=True)
    registered_pcd = registered_pcd + target_point_cloud

    o3d.visualization.draw_geometries([registered_pcd])
'''
    # pipline_point2plane_test(target_point_cloud, source_point_cloud)
    target_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_1.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)

    source_point_cloud = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/zivid 3d/pc_part_2.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)

    # _running_time()
    time.perf_counter()
    registered_pcd = zivid_3d_registration(target_point_cloud, source_point_cloud, visualization=True)
    print(time.perf_counter())

    visualization_point_cloud(source_point_cloud=registered_pcd,
                              target_point_cloud_with_background=target_point_cloud)
'''
