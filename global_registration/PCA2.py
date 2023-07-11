import open3d as o3d

import numpy as np

import copy

from sklearn.decomposition import PCA

mesh = o3d.io.read_triangle_mesh(r"C:\Users\Lenovo\Desktop\Alignment\Alignment\Motor_001.stl")
# http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html#open3d.geometry.TriangleMesh

pcd = mesh.sample_points_uniformly(number_of_points=10000)

translation_vector = pcd.get_center()

mesh.translate(-translation_vector, relative=True)

pcd.translate(-translation_vector, relative=True)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

# Perform PCA analysis

centered_pcd_np = np.array(pcd.points)

pca = PCA()

pca.fit(centered_pcd_np)

print('components:', pca.components_)

print('variance:', pca.explained_variance_ratio_)

pc_cloud = pca.transform(centered_pcd_np)

T = np.eye(4)

T[:3, :3] = pca.components_

T[0, 3] = 1

mesh.transform(T)

pcd_pca = o3d.geometry.PointCloud()

pcd_pca.points = o3d.utility.Vector3dVector(pc_cloud)

pcd_pca.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(90), 0]))

mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(90), 0]))

o3d.visualization.draw_geometries([pcd_pca, mesh_frame])

rot_z = "y"  # input("Rotate z-axis? Type y")

if rot_z == "y":
    pcd_pca.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(180), 0]))

    mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(180), 0]))

    o3d.visualization.draw_geometries([pcd_pca, mesh_frame, mesh])

    # o3d.io.write_triangle_mesh("/home/erik/Downloads/Motor_002_PCA.ply", mesh)
