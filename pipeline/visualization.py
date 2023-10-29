import numpy as np
import open3d as o3d



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


def get_arrow(begin=[0, 0, 0], vec=[0, 0, 1], save_dir=None):
    '''

    :param begin: beginning point of the arrow
    :param vec: vector from begin to end
    :return:
        mesh_frame:             coordinate axis
        mesh_arrow:             arrow
        mesh_sphere_begin:      sphere located at the beginning point
        mesh_sphere_end:        sphere located at the end point
    '''
    begin = begin
    end = np.add(begin, vec)
    vec_Arr = np.array(end) - np.array(begin)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])

    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * 1,
        cone_radius=0.06 * 1,
        cylinder_height=0.8 * 1,
        cylinder_radius=0.04 * 1
    )
    mesh_arrow.paint_uniform_color([1, 0, 0])
    mesh_arrow.compute_vertex_normals()

    mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=20)
    mesh_sphere_begin.translate(begin)
    mesh_sphere_begin.paint_uniform_color([0, 1, 1])
    mesh_sphere_begin.compute_vertex_normals()

    mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=20)
    mesh_sphere_end.translate(end)
    mesh_sphere_end.paint_uniform_color([0, 1, 1])
    mesh_sphere_end.compute_vertex_normals()

    rot_mat = _caculate_align_mat(vec_Arr*30)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))

    if save_dir is not None:
        o3d.io.write_triangle_mesh(save_dir, mesh_arrow + mesh_sphere_begin)

    return mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end


if __name__ == "__main__":
    pass
