import open3d as o3d


def read_one_mesh(file_dir=r"C:\Users\Lenovo\Desktop\Alignment\Alignment\Motor_001.stl", save_dir=None,
                  number_of_points=200000):
    """
    :param save_dir:
    :param number_of_points:
    :param file_dir: direction of *.stl file
    :return: point cloud
    """

    mesh = o3d.io.read_triangle_mesh(file_dir)

    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)  # 600000
    if save_dir is not None:
        pcd.paint_uniform_color([1, 1, 0])
        o3d.io.write_point_cloud(save_dir, pcd, write_ascii=True)

    return pcd


if __name__ == "__main__":
    pass
