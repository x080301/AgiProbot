import open3d as o3d
import numpy as np
from tqdm import tqdm

import utilities.data_visualization
from utilities.point_cloud_operation import get_normal_array


def _get_point_clouds_in_dict(pcd):
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)

    points_dict = {}

    for i in tqdm(range(colors.shape[0])):
        if str(colors[i] * 255) not in points_dict.keys():
            print(str(colors[i] * 255))
            points_dict[str(colors[i] * 255)] = points[i].reshape((1, 3))
        else:
            points_dict[str(colors[i] * 255)] = np.row_stack((points_dict[str(colors[i] * 255)], points[i]))

    print(points_dict.keys())

    return points_dict


def _test_get_mainhousing_cylinder_axis():
    main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
    normals = get_normal_array(main_housing_pcd)
    get_mainhousing_cylinder_axis(normals, visualization=True)


def _get_alpha_beta_gama_from_z_y(z, y):
    # 定义向量
    v1 = z
    v2 = y

    # 求x轴与z'轴正交且指向投影的方向
    v3 = np.cross(np.array([0, 0, 1]), v1)
    v3 /= np.linalg.norm(v3)

    # 求z轴与y'轴正交且指向投影的方向
    v4 = np.cross(v2, v3)
    v4 /= np.linalg.norm(v4)

    # 构造旋转矩阵
    R = np.array([v3, np.cross(v4, v3), v4]).T

    # 求解欧拉角
    theta_x = np.arctan2(-R[1, 2], R[2, 2])
    theta_y = np.arctan2(R[0, 2], np.sqrt(R[1, 2] ** 2 + R[2, 2] ** 2))
    theta_z = np.arctan2(-R[0, 1], R[0, 0])

    return theta_x, theta_y, theta_z


def get_mainhousing_cylinder_axis(normals, visualization=False):
    normals_pcd = o3d.geometry.PointCloud()
    normals_pcd.points = o3d.utility.Vector3dVector(normals)

    # RANSAC
    plane_model, inliers = normals_pcd.segment_plane(distance_threshold=0.005,
                                                     ransac_n=10,
                                                     num_iterations=1000)

    [a, b, c, d] = plane_model

    if visualization:
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = normals_pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1, 0, 0])
        outlier_cloud = normals_pcd.select_by_index(inliers, invert=True)
        outlier_cloud.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    cylinder_axis = np.array([a, b, c])
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

    return cylinder_axis


def get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd):
    solenoid_center = solenoid_pcd.get_center()
    main_housing_center = main_housing_pcd.get_center()
    connector_center = connector_pcd.get_center()

    # z axis:
    main_housing_normals = get_normal_array(main_housing_pcd)
    mainhousing_cylinder_axis = get_mainhousing_cylinder_axis(main_housing_normals)
    z_axis = np.sign(np.dot(mainhousing_cylinder_axis, connector_center - main_housing_center)) \
             * mainhousing_cylinder_axis

    # y axis:
    y_axis = solenoid_center - main_housing_center - np.dot(z_axis, solenoid_center - main_housing_center) * z_axis
    y_axis /= np.linalg.norm(y_axis)

    return z_axis, y_axis


def get_rotation_matrix(solenoid_pcd, main_housing_pcd, connector_pcd):
    z_axis, y_axis = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)

    theta_x, theta_y, theta_z = _get_alpha_beta_gama_from_z_y(z_axis, y_axis)
    pcd = o3d.geometry.PointCloud()

    return pcd.get_rotation_matrix_from_xyz((theta_x, theta_y, theta_z))


def _test_zy_axis():
    solenoid_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[255. 165.   0.].pcd',
                                           remove_nan_points=True, remove_infinite_points=True,
                                           print_progress=True)
    main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
    connector_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[102. 255. 102.].pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    z, y = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)

    segmented_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    _, mesh_arrow, mesh_sphere_begin, _ = utilities.data_visualization.get_arrow(segmented_pcd.get_center(), z * 150)
    z_arrow_list = [mesh_arrow, mesh_sphere_begin]

    _, mesh_arrow, mesh_sphere_begin, _ = utilities.data_visualization.get_arrow(segmented_pcd.get_center(), y * 150)
    y_arrow_list = [mesh_arrow, mesh_sphere_begin]

    # o3d.visualization.draw_geometries([segmented_pcd])
    o3d.visualization.draw_geometries([segmented_pcd] + z_arrow_list + y_arrow_list)


def calibration_z(z_axis, pcd):
    x, y, z = z_axis

    axis_angle = np.array([np.arctan2(y, z), -np.arctan2(x, z), 0])
    Rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(Rotation_matrix, center=pcd.get_center())

    return pcd, Rotation_matrix


def _test_calibration_z():
    solenoid_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[255. 165.   0.].pcd',
                                           remove_nan_points=True, remove_infinite_points=True,
                                           print_progress=True)
    main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
    connector_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[102. 255. 102.].pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    z, y = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)

    segmented_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    translation_vector = segmented_pcd.get_center()
    segmented_pcd.translate(-translation_vector, relative=True)

    segmented_pcd, _ = calibration_z(z, segmented_pcd)

    # o3d.visualization.draw_geometries([segmented_pcd])
    o3d.io.write_point_cloud('E:/datasets/agiprobot/calibration/results/segmented_pcd_z_calibrated.pcd', segmented_pcd,
                             write_ascii=True)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([segmented_pcd, mesh_frame])


def calibration_y(y_axis, pcd, rotation_matrix_z):
    y_axis = np.matmul(rotation_matrix_z, y_axis)
    x, y, _ = y_axis

    axis_angle = np.array([0, 0, np.arctan2(x, y)])
    Rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(Rotation_matrix, center=pcd.get_center())

    return pcd


def _test_calibration_y():
    solenoid_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[255. 165.   0.].pcd',
                                           remove_nan_points=True, remove_infinite_points=True,
                                           print_progress=True)
    main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
    connector_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[102. 255. 102.].pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    z, y = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)

    segmented_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    translation_vector = segmented_pcd.get_center()
    segmented_pcd.translate(-translation_vector, relative=True)

    segmented_pcd, rotation_matrix_z = calibration_z(z, segmented_pcd)

    segmented_pcd = calibration_y(y, segmented_pcd, rotation_matrix_z)

    o3d.io.write_point_cloud('E:/datasets/agiprobot/calibration/results/segmented_pcd_zy_calibrated.pcd', segmented_pcd,
                             write_ascii=True)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([segmented_pcd, mesh_frame])


def _test_calibration_pcd():
    import time

    solenoid_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[255. 165.   0.].pcd',
                                           remove_nan_points=True, remove_infinite_points=True,
                                           print_progress=True)
    main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
    connector_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[102. 255. 102.].pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    segmented_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    time.perf_counter()
    # ****************************************** #
    segmented_pcd = pointcloud_alignment(solenoid_pcd, main_housing_pcd, connector_pcd, segmented_pcd)
    # ****************************************** #
    print(time.perf_counter())

    o3d.io.write_point_cloud('E:/datasets/agiprobot/calibration/results/calibrated_pcd.pcd', segmented_pcd,
                             write_ascii=True)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([segmented_pcd, mesh_frame])


def pointcloud_alignment(solenoid_pcd, main_housing_pcd, connector_pcd, segmented_pcd):
    z, y = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)

    # translate the point cloud center to the origin
    translation_vector = segmented_pcd.get_center()
    segmented_pcd.translate(-translation_vector, relative=True)

    # Rotate point cloud so that the point cloud z axis and coordinate z axis coincide
    segmented_pcd, rotation_matrix_z = calibration_z(z, segmented_pcd)

    # Rotate point cloud so that the point cloud y axis and coordinate y axis coincide
    segmented_pcd = calibration_y(y, segmented_pcd, rotation_matrix_z)

    return segmented_pcd


if __name__ == "__main__":
    # _test_get_mainhousing_cylinder_axis()
    # _test_zy_axis()
    # _test_calibration_z()
    # _test_calibration_y()
    _test_calibration_pcd()

    pass
