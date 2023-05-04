import time
import copy
import open3d as o3d
import numpy as np
from utilities.data_visualization import visualization_point_cloud


def preprocess_point_cloud(pcd, voxel_size, visualization=False):
    if visualization:
        print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    if visualization:
        print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    if visualization:
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

    target = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/one_view_motor_only.pcd',
                                     remove_nan_points=True, remove_infinite_points=True,
                                     print_progress=True)
    source = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                     remove_nan_points=True, remove_infinite_points=True,
                                     print_progress=True)

    source = translation(target, source)

    '''trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)'''
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size, True)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, True)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
    return result


def raw_pipline():
    voxel_size = 3  # 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    draw_registration_result(source_down, target_down, result_fast.transformation)


def translation(target_point_cloud, source_point_cloud):
    translation_vector = target_point_cloud.get_center() - source_point_cloud.get_center()

    source_point_cloud.translate(translation_vector, relative=True)

    return source_point_cloud


def _rotation(source_point_cloud):
    rotated_point_cloud = copy.deepcopy(source_point_cloud)

    euler_angle = rotated_point_cloud.get_rotation_matrix_from_xyz((-np.pi * 30. / 180., np.pi * 3 / 4., 0))
    rotated_point_cloud.rotate(euler_angle)

    return rotated_point_cloud


def global_registration(target_point_cloud, source_point_cloud, visualization=False):
    if visualization:
        time.perf_counter()

    voxel_size = 3  # 0.05
    distance_threshold = voxel_size * 0.5

    # source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
    # print(":: Load two point clouds and disturb initial pose.")

    translation_vector = target_point_cloud.get_center() - source_point_cloud.get_center()
    source_point_cloud.translate(translation_vector, relative=True)

    # draw_registration_result(source_point_cloud, target_point_cloud, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source_point_cloud, voxel_size, visualization)
    target_down, target_fpfh = preprocess_point_cloud(target_point_cloud, voxel_size, visualization)

    # result_fast = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result_fast = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
    print(result_fast)
    # draw_registration_result(source_down, target_down, result_fast.transformation)
    source_point_cloud.transform(result_fast.transformation)

    if visualization:
        print(time.perf_counter())
        visualization_point_cloud(source_point_cloud=source_point_cloud,
                      target_point_cloud_with_background=target_point_cloud,
                      save=False)

    return source_point_cloud


if __name__ == "__main__":
    # raw_pipline()
    pass
