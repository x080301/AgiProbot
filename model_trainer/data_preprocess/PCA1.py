import open3d as o3d

import numpy as np

from sklearn.decomposition import PCA


def read_mesh(file_dir=r"C:\Users\Lenovo\Desktop\Alignment\Alignment\Motor_001.stl"):
    '''

    :param file_dir: direction of *.stl file
    :return: point cloud
    '''
    mesh = o3d.io.read_triangle_mesh(file_dir)

    pcd = mesh.sample_points_uniformly(number_of_points=600000)  # 10000

    return pcd


def implement_pca(pcd, visualization=False, save_dir=None):
    translation_vector = pcd.get_center()

    pcd.translate(-translation_vector, relative=True)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

    # Perform PCA analysis

    centered_pcd_np = np.array(pcd.points)

    pca = PCA()

    pca.fit(centered_pcd_np)

    print('components:', pca.components_)

    print('variance:', pca.explained_variance_ratio_)

    pc_cloud = pca.transform(centered_pcd_np)

    pcd_pca = o3d.geometry.PointCloud()

    pcd_pca.points = o3d.utility.Vector3dVector(pc_cloud)

    pcd_pca.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(90), 0]))

    # o3d.visualization.draw_geometries([pcd_pca, mesh_frame])

    pcd_pca.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(180), 0]))
    pcd_pca.paint_uniform_color([0.7, 0.7, 0.7])

    if save_dir is not None:
        o3d.io.write_point_cloud(save_dir, pcd_pca, write_ascii=True)
    if visualization:
        o3d.visualization.draw_geometries([pcd_pca, mesh_frame])

    return pcd_pca


if __name__ == "__main__":

    import os

    for file_name in os.listdir(r'C:\Users\Lenovo\Desktop\Alignment\Alignment'):
        if '.stl' in file_name:
            pcd = read_mesh(r"C:\Users\Lenovo\Desktop\Alignment\Alignment/" + file_name)
            implement_pca(pcd, visualization=True, save_dir=r'C:\Users\Lenovo\Desktop/' + file_name + '.pcd')
