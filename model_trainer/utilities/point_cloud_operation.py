import numpy as np
import open3d as o3d


def get_normal_array(pcd):
    radius = 1  # 5  # 1 # 0.5 # 0.1 # 0.01  # max search radius
    max_nn = 30  # max points in the search sphere
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    return np.array(pcd.normals)


if __name__ == "__main__":
    pass
