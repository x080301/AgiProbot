import copy

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def draw_3d_histogram(data3d, num_bins=10, title=None):
    # 将点云数据分成立方体
    hist, edges = np.histogramdd(data3d, bins=num_bins)
    hist = hist / np.max(hist) * 300.

    # 获取每个立方体的中心点
    x_centers = (edges[0][1:] + edges[0][:-1]) / 2
    y_centers = (edges[1][1:] + edges[1][:-1]) / 2
    z_centers = (edges[2][1:] + edges[2][:-1]) / 2

    # 创建一个三维坐标轴对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 在每个中心点绘制一个球形点云，大小为每个立方体内的点数
    for i in range(num_bins):
        for j in range(num_bins):
            for k in range(num_bins):
                center = (x_centers[i], y_centers[j], z_centers[k])
                size = hist[i, j, k]
                ax.scatter(*center, s=size, c='blue')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if title is not None:
        plt.title(title)

    # 显示图形
    plt.show()


# # 生成随机点云数据
# num_points = 1000
# x = np.random.normal(size=num_points)
# y = np.random.normal(size=num_points)
# z = np.random.normal(size=num_points)


# def draw_3dhisgram(x):


def _get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def _caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = _get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = _get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat


def get_arrow(begin=[0, 0, 0], vec=[0, 0, 1]):
    '''

    :param begin: beginning point of the arrow
    :param vec: vector from begin to end
    :return:
        mesh_frame:             coordinate axis
        mesh_arrow:             arrow
        mesh_sphere_begin:      sphere located at the beginning point
        mesh_sphere_end:        sphere located at the end point
    '''
    z_unit_Arr = np.array([0, 0, 1])
    begin = begin
    end = np.add(begin, vec)
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])

    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * 1,
        cone_radius=0.06 * 1,
        cylinder_height=0.8 * 1,
        cylinder_radius=0.04 * 1
    )
    mesh_arrow.paint_uniform_color([0, 1, 0])
    mesh_arrow.compute_vertex_normals()

    mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=10, resolution=20)
    mesh_sphere_begin.translate(begin)
    mesh_sphere_begin.paint_uniform_color([0, 1, 1])
    mesh_sphere_begin.compute_vertex_normals()

    mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=10, resolution=20)
    mesh_sphere_end.translate(end)
    mesh_sphere_end.paint_uniform_color([0, 1, 1])
    mesh_sphere_end.compute_vertex_normals()

    rot_mat = _caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end


def visualization_point_cloud(transformation=None, target_point_cloud_with_background=None, source_point_cloud=None,
                              save_dir=None):
    '''
        visualization(transformation=None, target_point_cloud=None, source_point_cloud=None)
        visualize target and transformed point cloud

        Args:
            transformation (numpy.ndarray, optimal): 4×4 transformationmatri. default:
                [[1. 0. 0. 0.]
                 [0. 1. 0. 0.]
                 [0. 0. 1. 0.]
                 [0. 0. 0. 1.]]
            target_point_cloud (o3d.geometry.PointCloud(), optimal):
            source_point_cloud (o3d.geometry.PointCloud(), optimal):
            save_dir (str, optimal): where to save the result

            transformation=None, target_point_cloud=None, source_point_cloud=None

        :return: None
        '''

    if transformation is None:
        transformation = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
    if target_point_cloud_with_background is None:
        target_point_cloud_with_background = o3d.io.read_point_cloud(
            'E:/datasets/agiprobot/registration/one_view_bin.pcd',
            remove_nan_points=True,
            remove_infinite_points=True,
            print_progress=True)
    if source_point_cloud is None:
        source_point_cloud = o3d.io.read_point_cloud('E:/datasets/agiprobot/binlabel/full_model.pcd',
                                                     remove_nan_points=True, remove_infinite_points=True,
                                                     print_progress=True)

    tarDraw = copy.deepcopy(target_point_cloud_with_background)
    tarDraw.paint_uniform_color([0, 1, 0])
    srcDraw = copy.deepcopy(source_point_cloud)
    srcDraw.paint_uniform_color([1, 1, 0])

    if save_dir is not None:
        o3d.io.write_point_cloud(save_dir, srcDraw + tarDraw, write_ascii=True)

    # tarDraw.paint_uniform_color([0, 1, 1])
    srcDraw.transform(transformation)
    o3d.visualization.draw_geometries([srcDraw, tarDraw])


def draw_box(center):
    length = 200
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=length, height=length, depth=0.1)
    # mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([1, 1, 0])
    # o3d.visualization.draw_geometries([mesh_box])
    mesh_box.translate([-0.5 * length, -0.5 * length, 0])
    mesh_box.translate(-center, relative=True)

    return mesh_box


if __name__ == "__main__":
    data3d = np.random.normal(size=(1000, 3))
    draw_3d_histogram(data3d, 10)
