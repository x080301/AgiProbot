import copy
import time

import numpy as np
import open3d as o3d

from data_visualization import visualization_point_cloud
from global_registration import global_registration
from registration import point2plane, point2point, CoarseRegistrationExceptin

from visualization import get_arrow

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


def get_point_cloud_for_a_class(labelled_pcd, class_index, label_rgb_dic=None, save_dir=None):
    if label_rgb_dic is None:
        label_rgb_dic = rgb_dic

    rgb_dic_values = list(label_rgb_dic.values())
    rgb_dic_values_sum = [sum(x) for x in rgb_dic_values]  # [621, 128, 497, 459, 401, 420, 510, 100, 623, 575]

    point_cloud = o3d.io.read_point_cloud(labelled_pcd, remove_nan_points=True,
                                          remove_infinite_points=True, print_progress=True)

    colors = np.asarray(point_cloud.colors)
    points = np.asarray(point_cloud.points)

    bolt_point_index = []

    rgb_sum = colors.sum(axis=1)
    rgb_sum = np.around(rgb_sum * 255)
    for j in range(points.shape[0]):
        rgb_sum_j = int(rgb_sum[j])
        label = rgb_dic_values_sum.index(rgb_sum_j)
        if label == class_index:
            bolt_point_index.append(j)

    the_class_only_pcd = o3d.geometry.PointCloud()
    the_class_only_pcd.points = o3d.utility.Vector3dVector(points[bolt_point_index, :])
    the_class_only_pcd.colors = o3d.utility.Vector3dVector(colors[bolt_point_index, :])

    if save_dir is not None:
        print(the_class_only_pcd)
        o3d.io.write_point_cloud(filename=save_dir + r'\bolt_only.pcd',
                                 pointcloud=the_class_only_pcd,
                                 write_ascii=True)

    return the_class_only_pcd


def _test_read_and_get_bolt_points():
    get_point_cloud_for_a_class(r'E:\datasets\agiprobot\pipeline_demo\6_full_model.pcd', class_index=4,
                                save_dir=r'E:\datasets\agiprobot\pipeline_demo')


def dbscan(point_cloud, save_dir=None):
    colors = np.asarray(point_cloud.colors)
    points = np.asarray(point_cloud.points)

    labels = np.asarray(point_cloud.cluster_dbscan(eps=5, min_points=100))

    num_bolts = labels.max() + 1

    bolt_pcd_list = []
    for i in range(-1, num_bolts):
        bolt_index = np.where(labels == i)[0]

        bolt_pcd = o3d.geometry.PointCloud()
        bolt_pcd.points = o3d.utility.Vector3dVector(points[np.where(labels == i)[0], :])
        bolt_pcd.colors = o3d.utility.Vector3dVector(colors[np.where(labels == i)[0], :])

        if save_dir is not None:
            print(bolt_pcd)
            o3d.io.write_point_cloud(filename=save_dir + '/bolt_only_' + str(i) + '.pcd',
                                     pointcloud=bolt_pcd,
                                     write_ascii=True)

        if i == -1:
            if len(bolt_index) == 0:
                noise = None
            else:
                noise = bolt_pcd
        else:
            bolt_pcd_list.append(bolt_pcd)

    return bolt_pcd_list, noise


def _test_sbscan():
    point_cloud = get_point_cloud_for_a_class(r'E:\datasets\agiprobot\pipeline_demo\6_full_model.pcd',
                                              class_index=4)
    # dbscan(point_cloud, r'E:\datasets\agiprobot\pipeline_demo')
    print(dbscan(point_cloud))


def find_center(bolt_pcd_list):
    center_list = []
    for bolt_pcd in bolt_pcd_list:
        center_list.append(bolt_pcd.get_center())
    return center_list


def _test_find_center():
    point_cloud = get_point_cloud_for_a_class(r'E:\datasets\agiprobot\pipeline_demo\6_full_model.pcd',
                                              class_index=4)
    bolt_pcd_list, _ = dbscan(point_cloud)
    print(len(find_center(bolt_pcd_list)))


def get_mainhousing_cylinder_axis(normals):
    normals_pcd = o3d.geometry.PointCloud()
    normals_pcd.points = o3d.utility.Vector3dVector(normals)

    # RANSAC
    plane_model, inliers = normals_pcd.segment_plane(distance_threshold=0.005,
                                                     ransac_n=10,
                                                     num_iterations=1000)

    [a, b, c, d] = plane_model

    cylinder_axis = np.array([a, b, c])
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

    return cylinder_axis


def get_normal_array(pcd):
    radius = 1  # 5  # 1 # 0.5 # 0.1 # 0.01  # max search radius
    max_nn = 30  # max points in the search sphere
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    return np.array(pcd.normals)


def get_z_axis(point_cloud_dir):
    main_housing_pcd = get_point_cloud_for_a_class(point_cloud_dir, class_index=7)
    connector_pcd = get_point_cloud_for_a_class(point_cloud_dir, class_index=3)

    main_housing_center = main_housing_pcd.get_center()
    connector_center = connector_pcd.get_center()

    # z axis:
    main_housing_normals = get_normal_array(main_housing_pcd)
    mainhousing_cylinder_axis = get_mainhousing_cylinder_axis(main_housing_normals)
    z_axis = np.sign(np.dot(mainhousing_cylinder_axis, connector_center - main_housing_center)) \
             * mainhousing_cylinder_axis

    return z_axis


def _test_get_z_axis():
    print(get_z_axis(r'E:\datasets\agiprobot\pipeline_demo\6_full_model.pcd'))


def _test_visualization():
    point_cloud = get_point_cloud_for_a_class(r'E:\datasets\agiprobot\pipeline_demo\6_full_model.pcd',
                                              class_index=4)
    bolt_pcd_list, _ = dbscan(point_cloud)

    bolt_position = find_center(bolt_pcd_list)[0]

    normal = get_z_axis(r'E:\datasets\agiprobot\pipeline_demo\6_full_model.pcd')

    get_arrow(begin=bolt_position, vec=normal, save_dir=r'E:\datasets\agiprobot\pipeline_demo\arrow.ply')


def get_transform_matrix(target_point_cloud_dir, source_point_cloud_dir, algorithm='point2plane_multi_step',
                         visualization=False):
    '''

    :param target_point_cloud:
    :param source_point_cloud:
    :param algorithm: (string, optimal) 'point2plane_multi_step' or 'point2point_multi_step'
    :param visualization:
    :return: registered point cloud
    '''

    source_point_cloud = o3d.io.read_point_cloud(source_point_cloud_dir, remove_nan_points=True,
                                                 remove_infinite_points=True, print_progress=True)
    target_point_cloud = get_point_cloud_for_a_class(target_point_cloud_dir, 0)

    if visualization:
        time.perf_counter()

    if algorithm == 'point2plane_multi_step':
        fine_registration = point2plane
    elif algorithm == 'point2point_multi_step':
        fine_registration = point2point

    # global registration and first fine registration
    for i in range(5):
        # coarse_registered = _coarse_registration_hard_coding(target_point_cloud, source_point_cloud)

        # translation_vector = target_point_cloud.get_center() - source_point_cloud.get_center()
        # source_point_cloud.translate(translation_vector, relative=True)
        # registered_point_cloud = copy.deepcopy(
        #     source_point_cloud)  # global_registration(target_point_cloud, copy.deepcopy(source_point_cloud))
        # euler_angle = registered_point_cloud.get_rotation_matrix_from_xyz((0, np.pi * 135. / 180, 0))
        #
        # registered_point_cloud.rotate(euler_angle)

        registered_point_cloud, transform_matrix = global_registration(target_point_cloud,
                                                                       copy.deepcopy(source_point_cloud))
        # print(transform_matrix)

        if visualization:
            print("coarse_registration")
            registered_point_cloud.paint_uniform_color([1, 1, 0])
            visualization_point_cloud(source_point_cloud=registered_point_cloud,
                                      target_point_cloud_with_background=target_point_cloud)

        # When the coarse registration give an unsuitable,
        # the correspindence set of the fine registration will be too small.
        # The size of correspindence set is checked here,
        # too small -> registration_algorithm() returns None
        registered_point_cloud, transform_matrix_fine = fine_registration(target_point_cloud,
                                                                          registered_point_cloud,
                                                                          max_correspondence_distance=5,
                                                                          evaluate_coarse_registraion_min_correspindence=100000)

        if registered_point_cloud is not None:
            transform_matrix = transform_matrix_fine @ transform_matrix
            break
    else:
        raise CoarseRegistrationExceptin

    if visualization:
        print(time.perf_counter())
        print("fine_registration")
        visualization_point_cloud(source_point_cloud=registered_point_cloud,
                                  target_point_cloud_with_background=target_point_cloud)

    # registered = registration_algorithm(target_point_cloud, registered, max_correspondence_distance=1)
    registered_point_cloud, transform_matrix_fine = fine_registration(target_point_cloud, registered_point_cloud,
                                                                      max_correspondence_distance=0.2)

    transform_matrix = transform_matrix_fine @ transform_matrix

    if visualization:
        print(time.perf_counter())
        print("fine_registration")
        visualization_point_cloud(source_point_cloud=registered_point_cloud,
                                  target_point_cloud_with_background=target_point_cloud)

    return registered_point_cloud, transform_matrix


def whole_pipeline():
    point_cloud = get_point_cloud_for_a_class(r'E:\datasets\agiprobot\pipeline_demo\2_full_model.pcd',
                                              class_index=4)
    bolt_pcd_list, _ = dbscan(point_cloud)
    normal = get_z_axis(r'E:\datasets\agiprobot\pipeline_demo\2_full_model.pcd')

    registered_point_cloud, transform_matrix = get_transform_matrix(
        r'E:\datasets\agiprobot\pipeline_demo\2_b_colored.pcd',
        r'E:\datasets\agiprobot\pipeline_demo\2_full_model.pcd')

    for position in find_center(bolt_pcd_list):
        print(position[2])
    sign = [-1, 1, -1, 1, -1, 1, -1, -1, -1]
    for i, bolt_position in enumerate(find_center(bolt_pcd_list)):
        _, mesh_arrow, mesh_sphere_begin, _ = get_arrow(begin=bolt_position, vec=sign[i] * normal, )
        arrow = mesh_arrow + mesh_sphere_begin
        arrow.transform(transform_matrix)

        o3d.io.write_triangle_mesh(r'E:\datasets\agiprobot\pipeline_demo\arrow_' + str(i) + '.ply', arrow)


if __name__ == "__main__":
    whole_pipeline()
