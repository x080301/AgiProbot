import numpy as np
import open3d as o3d


def get_normal_array(pcd):
    radius = 1  # 5  # 1 # 0.5 # 0.1 # 0.01  # max search radius
    max_nn = 30  # max points in the search sphere
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    return np.array(pcd.normals)


def read_mesh(file_dir=r"C:\Users\Lenovo\Desktop\Alignment\Alignment\Motor_001.stl",save_dir=None, number_of_points=200000):
    '''

    :param file_dir: direction of *.stl file
    :return: point cloud
    '''
    mesh = o3d.io.read_triangle_mesh(file_dir)

    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)  # 600000
    if save_dir is not None:
        pcd.paint_uniform_color([1, 1, 0])
        o3d.io.write_point_cloud(save_dir, pcd)

    return pcd


if __name__ == "__main__":
    read_mesh(r'C:\Users\Lenovo\Desktop\untitled.ply',
              r'C:\Users\Lenovo\Desktop\MotorClean.plz.pcd')
