import numpy as np
import open3d as o3d

from data_preprocess.alignment import get_zy_axis, alignment_z, alignment_y, pointcloud_alignment, \
    get_mainhousing_cylinder_axis
from utilities.point_cloud_operation import get_normal_array

import unittest


class TestAlignment(unittest.TestCase):

    @unittest.skip('pass')
    def test_calibration_z(self):
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

        translation_vector = segmented_pcd.get_center()
        segmented_pcd.translate(-translation_vector, relative=True)

        segmented_pcd, _ = alignment_z(z, segmented_pcd)

        # o3d.visualization.draw_geometries([segmented_pcd])
        o3d.io.write_point_cloud('E:/datasets/agiprobot/calibration/results/segmented_pcd_z_calibrated.pcd',
                                 segmented_pcd,
                                 write_ascii=True)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([segmented_pcd, mesh_frame])

    @unittest.skip('pass')
    def test_calibration_y(self):
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

        translation_vector = segmented_pcd.get_center()
        segmented_pcd.translate(-translation_vector, relative=True)

        segmented_pcd, rotation_matrix_z = alignment_z(z, segmented_pcd)

        segmented_pcd, _ = alignment_y(y, segmented_pcd, rotation_matrix_z)

        o3d.io.write_point_cloud('E:/datasets/agiprobot/calibration/results/segmented_pcd_zy_calibrated.pcd',
                                 segmented_pcd,
                                 write_ascii=True)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([segmented_pcd, mesh_frame])

    @unittest.skip('pass')
    def test_calibration_pcd(self):
        import time

        solenoid_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[255. 165.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
        main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                                   remove_nan_points=True, remove_infinite_points=True,
                                                   print_progress=True)
        connector_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[102. 255. 102.].pcd',
                                                remove_nan_points=True, remove_infinite_points=True,
                                                print_progress=True)

        segmented_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                                remove_nan_points=True, remove_infinite_points=True,
                                                print_progress=True)

        time.perf_counter()
        # ****************************************** #
        segmented_pcd = pointcloud_alignment(solenoid_pcd, main_housing_pcd, connector_pcd, segmented_pcd)
        # ****************************************** #
        print(time.perf_counter())

        o3d.io.write_point_cloud('E:/datasets/agiprobot/calibration/results/calibrated_pcd.pcd', segmented_pcd,
                                 write_ascii=True)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([segmented_pcd, mesh_frame])

    @unittest.skip('pass')
    def test_get_mainhousing_cylinder_axis(self):
        main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                                   remove_nan_points=True, remove_infinite_points=True,
                                                   print_progress=True)

        normals = get_normal_array(main_housing_pcd)
        get_mainhousing_cylinder_axis(normals, visualization=True)

    def test_draw_and_save_for_ppt(self):
        from utilities import data_visualization

        solenoid_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[255. 165.   0.].pcd',
                                               remove_nan_points=True, remove_infinite_points=True,
                                               print_progress=True)
        main_housing_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[  0. 100.   0.].pcd',
                                                   remove_nan_points=True, remove_infinite_points=True,
                                                   print_progress=True)
        connector_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/calibration/pcds/[102. 255. 102.].pcd',
                                                remove_nan_points=True, remove_infinite_points=True,
                                                print_progress=True)

        segmented_pcd = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                                remove_nan_points=True, remove_infinite_points=True,
                                                print_progress=True)

        '''z_axis, y_axis = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)

        x, y, z = z_axis

        axis_angle = np.array([np.arctan2(y, z), -np.arctan2(x, z), 0])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(-axis_angle)

        xoy_plane = data_visualization.draw_box(-main_housing_pcd.get_center())
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150, origin=[0, 0, 0])

        xoy_plane.rotate(rotation_matrix, center=xoy_plane.get_center())
        #o3d.visualization.draw_geometries([box, mesh_frame])

        o3d.visualization.draw_geometries([connector_pcd, main_housing_pcd, solenoid_pcd, xoy_plane])
        o3d.io.write_triangle_mesh("C:/Users/Lenovo/Desktop/xoy_plane_main_housing.ply", xoy_plane)'''

        '''_, mesh_arrow, mesh_sphere_begin, mesh_sphere_end = data_visualization.get_arrow(main_housing_pcd.get_center(),
                                                                                         connector_pcd.get_center() - main_housing_pcd.get_center())
        o3d.visualization.draw_geometries([mesh_arrow, mesh_sphere_begin, mesh_sphere_end, segmented_pcd])
        o3d.io.write_triangle_mesh("C:/Users/Lenovo/Desktop/arrow_z_0.ply",
                                   mesh_arrow + mesh_sphere_begin + mesh_sphere_end)'''

        '''z_axis, y_axis = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)
        _, z_mesh_arrow, _, _ = data_visualization.get_arrow(segmented_pcd.get_center(), z_axis * 150)
        z_mesh_arrow.paint_uniform_color([0, 0, 1])
        _, y_mesh_arrow, mesh_sphere_begin, _ = data_visualization.get_arrow(segmented_pcd.get_center(), y_axis * 150)
        y_mesh_arrow.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([mesh_sphere_begin, z_mesh_arrow, y_mesh_arrow, segmented_pcd])

        o3d.io.write_triangle_mesh("C:/Users/Lenovo/Desktop/arrow_z.ply", z_mesh_arrow)
        o3d.io.write_triangle_mesh("C:/Users/Lenovo/Desktop/arrow_y.ply", y_mesh_arrow)
        o3d.io.write_triangle_mesh("C:/Users/Lenovo/Desktop/o.ply", mesh_sphere_begin)'''

        z_axis, y_axis = get_zy_axis(solenoid_pcd, main_housing_pcd, connector_pcd)
        x_axis = np.cross(z_axis, y_axis)
        _, x_mesh_arrow, _, _ = data_visualization.get_arrow(segmented_pcd.get_center(), -x_axis * 150)
        x_mesh_arrow.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([x_mesh_arrow, segmented_pcd])
        o3d.io.write_triangle_mesh("C:/Users/Lenovo/Desktop/arrow_x.ply", x_mesh_arrow)



