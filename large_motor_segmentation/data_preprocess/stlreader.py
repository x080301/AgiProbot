import open3d as o3d
import os


def read_mesh(file_dir=r"C:\Users\Lenovo\Desktop\Alignment\Alignment\Motor_001.stl", visualization=False, save_dir=None,
              number_of_points=200000):
    """

    :param save_dir:
    :param number_of_points:
    :param visualization:
    :param file_dir: direction of *.stl file
    :return: point cloud
    """
    mesh = o3d.io.read_triangle_mesh(file_dir)

    point_cloud = mesh.sample_points_uniformly(number_of_points=number_of_points)  # 600000
    if save_dir is not None:
        point_cloud.paint_uniform_color([1, 1, 0])
        o3d.io.write_point_cloud(save_dir, point_cloud)

    if visualization:
        o3d.visualization.draw_geometries([point_cloud])

    return point_cloud


def read_mesh_in_folder(file_folder, save_folder, number_of_points=200000):
    for root, _, files in os.walk(file_folder):
        for file_name in files:
            if '.stl' in file_name:
                save_dir = os.path.join(save_folder, root.split('\\')[-1] + '_' + file_name.split('.')[0] + '.pcd')

                read_mesh(os.path.join(root, file_name), save_dir=save_dir, number_of_points=number_of_points)


if __name__ == "__main__":
    read_mesh_in_folder(r'E:\datasets\agiprobot\fromJan\raw_data_18', r'C:\Users\Lenovo\Desktop\test',
                        number_of_points=600000)
