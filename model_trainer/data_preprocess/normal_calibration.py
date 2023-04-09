import open3d as o3d
import numpy as np
from tqdm import tqdm


def get_point_clouds_in_dict(pcd):
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


def get_normal_array(pcd):
    radius = 1  # 5  # 1 # 0.5 # 0.1 # 0.01  # max search radius
    max_nn = 30  # max points in the search sphere
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    return np.array(pcd.normals)


def test_get_mainhousing_cylinder_axis():
    main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
    normals = get_normal_array(main_housing_pcd)
    get_mainhousing_cylinder_axis(normals, visualization=True)


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


def get_alpha_beta_gama_from_z_y(z, y):
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

    theta_x, theta_y, theta_z = get_alpha_beta_gama_from_z_y(z_axis, y_axis)
    pcd = o3d.geometry.PointCloud()

    return pcd.get_rotation_matrix_from_xyz((theta_x, theta_y, theta_z))


def test_zy_axis():
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
    from utilities import draw

    _, mesh_arrow, mesh_sphere_begin, _ = draw.get_arrow(segmented_pcd.get_center(), z * 150)
    z_arrow_list = [mesh_arrow, mesh_sphere_begin]

    _, mesh_arrow, mesh_sphere_begin, _ = draw.get_arrow(segmented_pcd.get_center(), y * 150)
    y_arrow_list = [mesh_arrow, mesh_sphere_begin]

    # o3d.visualization.draw_geometries([segmented_pcd])
    o3d.visualization.draw_geometries([segmented_pcd] + z_arrow_list + y_arrow_list)


if __name__ == "__main__":
    # test_zy_axis()  # TODO: 展示2
    # test_get_mainhousing_cylinder_axis()  # TODO: 展示1

    '''
    import os
    from utilities import data_processing
    pcd = o3d.io.read_point_cloud(r'E:\datasets\agiprobot\Video snippets\full_model.pcd',
                                  remove_nan_points=True, remove_infinite_points=True,
                                  print_progress=True)

    points_dict = get_point_clouds_in_dict(pcd)
    for key_name in points_dict.keys():
        pcd_one_type = o3d.geometry.PointCloud()
        pcd_one_type.points = o3d.utility.Vector3dVector(points_dict[key_name])
        pcd_one_type.paint_uniform_color([0.7, 0.7, 0.7])

        o3d.io.write_point_cloud('C:/Users/Lenovo/Desktop/' + key_name + '.pcd', pcd_one_type, write_ascii=True)
    '''

    '''
    files_direction = 'E:/datasets/agiprobot/calibration/'
    for file_name in os.listdir(files_direction):
        pcd = o3d.io.read_point_cloud(files_direction + file_name,
                                      remove_nan_points=True, remove_infinite_points=True,
                                      print_progress=True)

        dict = {'[  0. 100.   0.].pcd': 'Main Housing',
                '[102. 140. 255.].pcd': 'Gear',
                '[102. 255. 102.].pcd': 'Connector',
                '[247.  77.  77.].pcd': 'Screws',
                '[255. 165.   0.].pcd': 'Solenoid',
                '[255. 255.   0.].pcd': 'Electrical Connector'
                }

        normals = get_normal_array(pcd)

        data_processing.draw_3d_histogram(normals, title=dict[file_name])
        '''

    '''
    pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                  remove_nan_points=True, remove_infinite_points=True,
                                  print_progress=True)

    normals = get_normal_array(pcd)
    get_mainhousing_cylinder_axis(normals,visualization=True)

    segmented_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1.0, cone_radius=1.5, cylinder_height=100.0,
                                                   cone_height=4.0, resolution=20, cylinder_split=4, cone_split=1)
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([segmented_pcd, arrow])#([arrow])#([segmented_pcd, arrow])

    # data_processing.draw_3d_histogram(normals, title='Main Housing')
'''

    '''
    solenoid_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[255. 165.   0.].pcd',
                                           remove_nan_points=True, remove_infinite_points=True,
                                           print_progress=True)
    main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
    connector_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[102. 255. 102.].pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    R = get_rotation_matrix(solenoid_pcd, main_housing_pcd, connector_pcd)

    segmented_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)
    import copy

    segmented_pcd_0 = copy.deepcopy(segmented_pcd)
    segmented_pcd.rotate(R, center=segmented_pcd.get_center())
    print(R)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([segmented_pcd_0, segmented_pcd, mesh_frame])
'''

    '''
    solenoid_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[255. 165.   0.].pcd',
                                           remove_nan_points=True, remove_infinite_points=True,
                                           print_progress=True)
    main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
    connector_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[102. 255. 102.].pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)

    top_normal, right_normal = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)

    axis = np.cross(top_normal, right_normal)  # 旋转轴
    angle = np.arccos(np.dot(top_normal, [0, 0, 1]))  # 旋转角度
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis, angle)

    # 应用旋转矩阵到点云
    segmented_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                            remove_nan_points=True, remove_infinite_points=True,
                                            print_progress=True)
    import copy

    segmented_pcd_0 = copy.deepcopy(segmented_pcd)
    segmented_pcd.rotate(R, segmented_pcd.get_center())

    # 显示结果
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([segmented_pcd_0, segmented_pcd, mesh_frame])
    '''

    pass
