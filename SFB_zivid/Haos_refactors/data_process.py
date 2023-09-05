import time
import os
import open3d as o3d
import numpy as np
import copy
import platform
import torch
import einops


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
                            visualization=False,
                            save_dir=None,
                            voxel_size=0.1):
    if pcd_directory not in registrted_folder:
        registrted_folder.append(pcd_directory)

        files = os.listdir(pcd_directory)

        files = [item for item in files if 'pc_part_' in item]
        files = [item for item in files if os.path.isfile(os.path.join(pcd_directory, item))]

        if len(files) > 0:
            registrate_one_folder(files, pcd_directory, save_dir, visualization, voxel_size)

    for root, _, files in os.walk(pcd_directory):

        if root in registrted_folder:
            continue
        else:
            registrted_folder.append(root)

        if len(files) == 0:
            continue

        registrate_one_folder(files, root, save_dir, visualization, voxel_size)

    return registrted_folder


def registrate_one_folder(files, root, save_dir, visualization, voxel_size):
    print(root)
    combined_point_cloud = None
    files.sort()
    for file_name in files:
        if 'pc_part_' in file_name:
            file_direction = os.path.join(root, file_name)
            target_point_cloud = o3d.io.read_point_cloud(file_direction,
                                                         remove_nan_points=True,
                                                         remove_infinite_points=True,
                                                         print_progress=True)

            # target_point_cloud, _ = target_point_cloud.remove_radius_outlier(nb_points=4, radius=0.5)
            if 'b' in root:
                # print('Do remove')
                target_point_cloud, _ = remove_black_noise_line(target_point_cloud, nb_points=150, radius=1.5)

            target_point_cloud = target_point_cloud.voxel_down_sample(voxel_size=voxel_size)
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

                combined_point_cloud = combined_point_cloud.voxel_down_sample(voxel_size=voxel_size)
    if visualization:
        o3d.visualization.draw_geometries([combined_point_cloud])
    if save_dir == None:
        print('save combined point cloud at:' + os.path.join(root, 'combined.pcd'))
        o3d.io.write_point_cloud(filename=os.path.join(root, 'combined.pcd'),
                                 pointcloud=combined_point_cloud)
    else:
        print('save combined point cloud at:' + os.path.join(save_dir,
                                                             root.split('/')[-1].split('\\')[-1] + '_combined.pcd'))
        o3d.io.write_point_cloud(filename=os.path.join(save_dir,
                                                       root.split('/')[-1].split('\\')[-1] + '_combined.pcd'),
                                 pointcloud=combined_point_cloud)


def remove_black_noise_line_knn(source_pcd, k):
    """

    :return: cleaned_pcd,outlier_pcd
    """
    points = torch.asarray(np.asarray(source_pcd.points))
    colors = torch.asarray(np.asarray(source_pcd.colors))

    average_color_pcd = source_pcd  # o3d.geometry.PointCloud()
    average_color_in_neighborhood = get_average_color_in_neighborhood(points, colors, k)

    average_color_pcd.colors = o3d.utility.Vector3dVector(average_color_in_neighborhood)

    return average_color_pcd


def _test_remove_black_noise_line_knn():
    source_pcd = o3d.io.read_point_cloud(r'E:\SFB_Demo\models\clean\17b\pc_part_1.pcd',
                                         remove_nan_points=True,
                                         remove_infinite_points=True,
                                         print_progress=True)

    o3d.io.write_point_cloud(filename=r'E:\SFB_Demo\models\clean\pc_part_1_knn.pcd',
                             pointcloud=remove_black_noise_line_knn(source_pcd, 5), write_ascii=True)

    source_pcd = o3d.io.read_point_cloud(r'E:\SFB_Demo\models\clean\17b_combined.pcd',
                                         remove_nan_points=True,
                                         remove_infinite_points=True,
                                         print_progress=True)

    o3d.io.write_point_cloud(filename=r'E:\SFB_Demo\models\clean\17b_combined_knn.pcd',
                             pointcloud=remove_black_noise_line_knn(source_pcd, 5), write_ascii=True)


def get_average_color_in_neighborhood(points, colors, k):
    """

    :param points: [N, 3]
    :param colors: [N, 3]
    :param k:
    :return:average color:[N,3]
    """
    colors = einops.rearrange(colors, 'n c -> 1 c n')
    points = einops.rearrange(points, 'n c -> 1 c n')

    idx = knn(points, k)  # batch_size x num_points x 20
    average_in_neighborhood = index_points_neighbors(colors, idx)  # _    (B,N,C) -> (B,N,C)

    average_in_neighborhood = einops.rearrange(average_in_neighborhood, '1 c n -> n c')

    return average_in_neighborhood


def knn(x, k):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        idx: sample index data, [B, N, K]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B,N,N)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def index_points_neighbors(x, idx):
    """
    Input:
        points: input points data, [B, C, N]
        idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    x = x.transpose(2, 1).contiguous()

    batch_size = x.size(0)
    num_points = x.size(1)
    num_dims = x.size(2)

    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, -1, num_dims)

    average_in_neighborhood = torch.mean(neighbors, dim=2)  # [B, N, K, C]->[B, N, C]

    return average_in_neighborhood


def _remove_black_noise_line(source_pcd, threshold=0.9):
    """

    :return: cleaned_pcd,outlier_pcd
    """

    cleaned_pcd = o3d.geometry.PointCloud()
    outlier_pcd = o3d.geometry.PointCloud()

    points = torch.asarray(np.asarray(source_pcd.points))
    colors = torch.asarray(np.asarray(source_pcd.colors))

    cleaned_index = torch.sum(colors, dim=1) > threshold
    cleaned_pcd.points = o3d.utility.Vector3dVector(points[cleaned_index, :])
    cleaned_pcd.colors = o3d.utility.Vector3dVector(colors[cleaned_index, :])

    outlier_index = ~cleaned_index
    outlier_pcd.points = o3d.utility.Vector3dVector(points[outlier_index, :])
    outlier_pcd.colors = o3d.utility.Vector3dVector(colors[outlier_index, :])
    # target_point_cloud, _ = target_point_cloud.remove_radius_outlier(nb_points=4, radius=0.5)

    return cleaned_pcd, outlier_pcd


def remove_black_noise_line(source_pcd, threshold=0.5, nb_points=150, radius=1.5):
    """

    :return: cleaned_pcd,outlier_pcd
    """

    lighter_pcd = o3d.geometry.PointCloud()
    darker_pcd = o3d.geometry.PointCloud()

    points = torch.asarray(np.asarray(source_pcd.points))
    colors = torch.asarray(np.asarray(source_pcd.colors))

    lighter_index = torch.sum(colors, dim=1) > threshold
    lighter_pcd.points = o3d.utility.Vector3dVector(points[lighter_index, :])
    lighter_pcd.colors = o3d.utility.Vector3dVector(colors[lighter_index, :])

    darker_index = ~lighter_index
    darker_pcd.points = o3d.utility.Vector3dVector(points[darker_index, :])
    darker_pcd.colors = o3d.utility.Vector3dVector(colors[darker_index, :])

    _, inlier_inex = darker_pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_pcd = lighter_pcd + darker_pcd.select_by_index(inlier_inex)
    outlier_pcd = darker_pcd.select_by_index(inlier_inex, invert=True)

    return inlier_pcd, outlier_pcd


def _test_remove_black_noise_line():
    source_pcd = o3d.io.read_point_cloud(r'E:\SFB_Demo\models\clean\17b\pc_part_1.pcd',
                                         remove_nan_points=True,
                                         remove_infinite_points=True,
                                         print_progress=True)

    # cleaned_pcd, outlier_pcd = remove_black_noise_line(source_pcd, nb_points=100, radius=1.5)
    cleaned_pcd, outlier_pcd = remove_black_noise_line(source_pcd, nb_points=150, radius=1.5)

    o3d.io.write_point_cloud(filename=r'E:\SFB_Demo\models\clean\pc_part_1_cleaned.pcd', pointcloud=cleaned_pcd,
                             write_ascii=True)
    o3d.io.write_point_cloud(filename=r'E:\SFB_Demo\models\clean\pc_part_1_outlier.pcd', pointcloud=outlier_pcd,
                             write_ascii=True)

    # source_pcd = o3d.io.read_point_cloud(r'E:\SFB_Demo\models\clean\17b_combined.pcd',
    #                                      remove_nan_points=True,
    #                                      remove_infinite_points=True,
    #                                      print_progress=True)
    #
    # cleaned_pcd, outlier_pcd = remove_black_noise_line(source_pcd, nb_points=100, radius=2)
    #
    # o3d.io.write_point_cloud(filename=r'E:\SFB_Demo\models\clean\17b_combined_cleaned.pcd', pointcloud=cleaned_pcd,
    #                          write_ascii=True)
    # o3d.io.write_point_cloud(filename=r'E:\SFB_Demo\models\clean\17b_combined_outlier.pcd', pointcloud=outlier_pcd,
    #                          write_ascii=True)


def _pipeline_registration():
    system_type = platform.system().lower()  # 'windows' or 'linux'
    if system_type == 'windows':
        save_dir = 'E:/SFB_zivid/SFB_Demo/models/'
    else:
        save_dir = '/home/wbk-ur2/dual_ws/src/agiprobot_control/scripts/SFB_Demo/models/'

    pcd_directory = save_dir + 'scan_1'

    registration_in_folders(pcd_directory='E:/SFB_Demo/models/scan_2',
                            save_dir='E:/SFB_Demo/models/registered')
    registration_in_folders(pcd_directory='E:/SFB_Demo/models/scan_3',
                            save_dir='E:/SFB_Demo/models/registered')


if __name__ == "__main__":
    # _test_remove_black_noise_line()
    # _test_remove_black_noise_line_knn()
    _pipeline_registration()
