import open3d as o3d
import numpy as np
from tqdm import tqdm
import cv2

import data_visualization
from point_cloud_operation import get_normal_array


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


def get_a_point_on_cylinder_axis(z_axis, point_cloud, visualization=False):
    # Rotate point cloud so that the point cloud z axis and coordinate z axis coincide
    segmented_pcd, rotation_matrix_z = alignment_z(z_axis, point_cloud)

    #
    img = np.array(segmented_pcd.points)[:, 0:2]
    print(type(img))
    print(img)
    if visualization:
        cv2.imshow(img)


def get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd, point_cloud_center_on_zylinder_axis=True):
    solenoid_center = solenoid_pcd.get_center()
    main_housing_center = main_housing_pcd.get_center()
    connector_center = connector_pcd.get_center()

    # z axis:
    main_housing_normals = get_normal_array(main_housing_pcd)
    mainhousing_cylinder_axis = get_mainhousing_cylinder_axis(main_housing_normals)
    z_axis = np.sign(np.dot(mainhousing_cylinder_axis, connector_center - main_housing_center)) \
             * mainhousing_cylinder_axis

    # y axis:

    if not point_cloud_center_on_zylinder_axis:
        solenoid_center = get_a_point_on_cylinder_axis(z_axis, solenoid_pcd, visualization=True)
        main_housing_center = get_a_point_on_cylinder_axis(z_axis, main_housing_pcd)

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

    _, mesh_arrow, mesh_sphere_begin, _ = data_visualization.get_arrow(segmented_pcd.get_center(), z * 150)
    z_arrow_list = [mesh_arrow, mesh_sphere_begin]

    _, mesh_arrow, mesh_sphere_begin, _ = data_visualization.get_arrow(segmented_pcd.get_center(), y * 150)
    y_arrow_list = [mesh_arrow, mesh_sphere_begin]

    # o3d.visualization.draw_geometries([segmented_pcd])
    o3d.visualization.draw_geometries([segmented_pcd] + z_arrow_list + y_arrow_list)


def alignment_z(z_axis, pcd):
    x, y, z = z_axis

    axis_angle = np.array([np.arctan2(y, z), -np.arctan2(x, z), 0])
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(rotation_matrix, center=pcd.get_center())

    return pcd, rotation_matrix


def alignment_y(y_axis, pcd, rotation_matrix_z):
    y_axis = np.matmul(rotation_matrix_z, y_axis)
    x, y, _ = y_axis

    axis_angle = np.array([0, 0, np.arctan2(x, y)])
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(rotation_matrix, center=pcd.get_center())

    return pcd, rotation_matrix


def cylinder_and_its_normal():
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=100, height=400)
    pcd = mesh_cylinder.sample_points_uniformly(number_of_points=500000)

    pcd.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd])

    normals = get_normal_array(pcd)
    normals_pcd = o3d.geometry.PointCloud()
    normals_pcd.points = o3d.utility.Vector3dVector(normals)
    # RANSAC
    plane_model, inliers = normals_pcd.segment_plane(distance_threshold=0.005,
                                                     ransac_n=10,
                                                     num_iterations=1000)

    inlier_cloud = normals_pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud = normals_pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def get_mainhousing_cylinder_axis(normals, pcd=None, visualization=False):
    normals_pcd = o3d.geometry.PointCloud()
    normals_pcd.points = o3d.utility.Vector3dVector(normals)

    # RANSAC
    plane_model, inliers = normals_pcd.segment_plane(distance_threshold=0.005,
                                                     ransac_n=10,
                                                     num_iterations=1000)

    [a, b, c, d] = plane_model

    cylinder_axis = np.array([a, b, c])
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

    if visualization:
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        if pcd is not None:
            print("in")
            inlier_cloud = pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1, 0, 0])  # ([0, 1, 0])#([1, 0, 0])
            outlier_cloud = pcd.select_by_index(inliers, invert=True)
            outlier_cloud.paint_uniform_color([0, 1, 0])
            # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

            vis = o3d.visualization.Visualizer()
            vis.create_window()  # 创建窗口
            render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
            render_option.point_size = 4  # 2.0  # 设置渲染点的大小
            vis.add_geometry(inlier_cloud)  # 添加点云
            vis.add_geometry(outlier_cloud)
            vis.run()
        else:
            inlier_cloud = normals_pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1, 0, 0])  # ([0, 1, 0])#([1, 0, 0])
            outlier_cloud = normals_pcd.select_by_index(inliers, invert=True)
            outlier_cloud.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

            vis = o3d.visualization.Visualizer()
            vis.create_window()  # 创建窗口
            render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
            render_option.point_size = 2.0  # 设置渲染点的大小
            vis.add_geometry(inlier_cloud)  # 添加点云
            vis.add_geometry(outlier_cloud)
            vis.run()

    return cylinder_axis


def point_cloud_alignment(solenoid_pcd, main_housing_pcd, connector_pcd, segmented_pcd):
    # get z and y axis of the point cloud
    z_axis, y_axis = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)

    # translate the point cloud center to the origin
    translation_vector = segmented_pcd.get_center()
    segmented_pcd.translate(-translation_vector, relative=True)

    # Rotate point cloud so that the point cloud z axis and coordinate z axis coincide
    segmented_pcd, rotation_matrix_z = alignment_z(z_axis, segmented_pcd)

    # Rotate point cloud so that the point cloud y axis and coordinate y axis coincide
    segmented_pcd, _ = alignment_y(y_axis, segmented_pcd, rotation_matrix_z)

    return segmented_pcd


def read_labeled_pcd_as_multi_pcd(direction):
    points = []
    solenoid_points = []
    main_housing_points = []
    connector_points = []
    with open(direction, 'r') as f:

        head_flag = True
        while True:
            # for i in range(12):
            oneline = f.readline()

            if head_flag:
                if 'DATA ascii' in oneline:
                    head_flag = False
                    continue
                else:
                    continue

            if not oneline:
                break

            x, y, z, _, label, _ = list(oneline.strip('\n').split(' '))  # '0 0 0 1646617 8 -1\n'

            if x == '0' and y == '0' and z == '0':
                continue
            x, y, z, label = float(x), float(y), float(z), int(label)
            points.append(np.array([x, y, z]))
            if label == 3:
                connector_points.append(np.array([x, y, z]))


            elif label == 5:
                solenoid_points.append(np.array([x, y, z]))
            elif label == 7:
                main_housing_points.append(np.array([x, y, z]))

    solenoid_point_cloud = o3d.geometry.PointCloud()
    solenoid_point_cloud.points = o3d.utility.Vector3dVector(solenoid_points)

    main_housing_point_cloud = o3d.geometry.PointCloud()
    main_housing_point_cloud.points = o3d.utility.Vector3dVector(main_housing_points)

    connector_point_cloud = o3d.geometry.PointCloud()
    connector_point_cloud.points = o3d.utility.Vector3dVector(connector_points)

    whole_point_cloud = o3d.geometry.PointCloud()
    whole_point_cloud.points = o3d.utility.Vector3dVector(points)

    return solenoid_point_cloud, main_housing_point_cloud, connector_point_cloud, whole_point_cloud


class PointCloudAlignment:
    def __init__(self, solenoid_pcd, main_housing_pcd, connector_pcd):
        pass

    def get_xyz_axis(self):
        pass

    def visualization(self):
        pass

    def get_alignmented_pcd(self, pcd):
        pass

    def get_rotation_matrix(self):
        pass


def _pipeline_demo_registration_for_zivid_3d_pcd():
    source_point_cloud = o3d.io.read_point_cloud(r'E:\SFB_Demo\models\scan_2\18t_combined.pcd',
                                                 remove_nan_points=True,
                                                 remove_infinite_points=True,
                                                 print_progress=True)
    solenoid_pcd, main_housing_pcd, connector_pcd, _ = read_labeled_pcd_as_multi_pcd(
        r'E:\SFB_Demo\models\scan_3\test\18t_labeled.pcd')

    # get z and y axis of the point cloud
    z_axis, y_axis = get_zy_axis(solenoid_pcd, main_housing_pcd + solenoid_pcd, connector_pcd)
    # z_axis = -z_axis

    # translate the point cloud center to the origin
    translation_vector = source_point_cloud.get_center()
    source_point_cloud.translate(-translation_vector, relative=True)

    # Rotate point cloud so that the point cloud z axis and coordinate z axis coincide
    source_point_cloud, rotation_matrix_z = alignment_z(z_axis, source_point_cloud)

    source_point_cloud.translate(translation_vector, relative=True)

    o3d.io.write_point_cloud(filename=r'E:\SFB_Demo\models\scan_3\test\18t_alignment.pcd',
                             pointcloud=source_point_cloud)


if __name__ == "__main__":
    _pipeline_demo_registration_for_zivid_3d_pcd()
