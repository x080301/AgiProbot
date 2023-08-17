#! /usr/bin/env python3

from cmath import nan
import rospy
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
import tf2_msgs.msg
import numpy as np
import datetime
import copy
from util import *
import open3d as o3d
from pathlib import Path
import time
import os

'''Creates RGB Model only. Segmentation is skipped. Is faster when only wanting a RGB model'''


def create_open3d_point_cloud(pc_global):
    xyz = pc_global[:, :3]
    rgb = pc_global[:, 3:]

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    point_cloud_open3d.colors = o3d.utility.Vector3dVector(rgb)

    refined_point_cloud_open3d = o3d.geometry.PointCloud.remove_non_finite_points(point_cloud_open3d, remove_nan=True,
                                                                                  remove_infinite=True)

    return refined_point_cloud_open3d


def remove_outlier_pc(pc):
    pc, ind = o3d.geometry.PointCloud.remove_radius_outlier(pc, 10, 2)
    pc, ind = o3d.geometry.PointCloud.remove_statistical_outlier(pc, 10, 2)
    return pc


def point2point(target_point_cloud, source_point_cloud, max_correspondence_distance=10):
    pipreg = o3d.pipelines.registration

    reg = pipreg.registration_icp(source_point_cloud, target_point_cloud,
                                  max_correspondence_distance=max_correspondence_distance,
                                  estimation_method=
                                  # pipreg.TransformationEstimationPointToPlane())
                                  # pipreg.TransformationEstimationForGeneralizedICP())
                                  # pipreg.TransformationEstimationForColoredICP())
                                  pipreg.TransformationEstimationPointToPoint())

    print(reg.transformation)
    print(reg)
    print(type(reg.transformation))

    result_point_cloud = source_point_cloud  # copy.deepcopy(source_point_cloud)
    result_point_cloud = result_point_cloud.transform(reg.transformation)

    return result_point_cloud


def point2plane(target_point_cloud, source_point_cloud, max_correspondence_distance=10,
                evaluate_coarse_registraion_min_correspindence=None):
    radius = 1  # 5  # 1 # 0.5 # 0.1 # 0.01  # max search radius
    max_nn = 30  # max points in the search sphere
    source_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    pipreg = o3d.pipelines.registration
    # target and source are swapped, since target is full model and computation of its normal is easier.
    reg = pipreg.registration_icp(target_point_cloud, source_point_cloud,
                                  max_correspondence_distance=max_correspondence_distance,
                                  estimation_method=
                                  pipreg.TransformationEstimationPointToPlane()
                                  # pipreg.TransformationEstimationForGeneralizedICP()
                                  # pipreg.TransformationEstimationForColoredICP()
                                  # pipreg.TransformationEstimationPointToPoint()
                                  )

    print(reg)

    if evaluate_coarse_registraion_min_correspindence is not None \
            and np.array(reg.correspondence_set).shape[0] < evaluate_coarse_registraion_min_correspindence:

        return None

    else:

        result_point_cloud = source_point_cloud  # copy.deepcopy(source_point_cloud)
        # result_point_cloud = result_point_cloud.transform(transformation_matrix)

        # Transform
        # https://de.mathworks.com/matlabcentral/answers/321703-how-can-i-calculate-the-reverse-of-a-rotation-and-translation-that-maps-a-cloud-of-points-c1-to-an
        rotation_matrix = np.asarray(reg.transformation[0:3, 0:3])
        translation_vector = reg.transformation[0:3, 3]

        result_points = np.asarray(result_point_cloud.points)
        result_points = (result_points - translation_vector).reshape((-1, 3, 1))

        # print(np.linalg.inv(rotation_matrix).shape)
        # print(result_points.shape)
        result_points = (np.linalg.inv(rotation_matrix) @ result_points).reshape((-1, 3))

        result_point_cloud.points = o3d.utility.Vector3dVector(result_points)

        return result_point_cloud


class CoarseRegistrationExceptin(Exception):
    "this is user's Exception for unsuccessful coarse registration "

    def __init__(self):
        pass

    def __str__(self):
        print("coarse registration failed")


def zivid_3d_registration(target_point_cloud, source_point_cloud, rotation_per_capture,
                          algorithm='point2plane_multi_step',
                          visualization=False):
    '''

    :param target_point_cloud:
    :param source_point_cloud:
    :param algorithm: (string, optimal) 'point2plane_multi_step' or 'point2point_multi_step'
    :param visualization:
    :return: registered point cloud
    '''

    if visualization:
        time.perf_counter()
    # target_point_cloud = get_motor_only_pcd(target_point_cloud)

    if algorithm == 'point2plane_multi_step':
        fine_registration = point2plane
    elif algorithm == 'point2point_multi_step':
        fine_registration = point2point

    # global registration and first fine registration
    # coarse_registered = _coarse_registration_hard_coding(target_point_cloud, source_point_cloud)

    registered_point_cloud = copy.deepcopy(
        source_point_cloud)  # global_registration(target_point_cloud, copy.deepcopy(source_point_cloud))
    euler_angle = registered_point_cloud.get_rotation_matrix_from_xyz((0, 0, np.pi * rotation_per_capture / 180))
    registered_point_cloud.rotate(euler_angle)

    if visualization:
        registered_point_cloud_v = copy.deepcopy(registered_point_cloud)
        registered_point_cloud_v.paint_uniform_color([1, 1, 0])
        target_point_cloud_v = copy.deepcopy(target_point_cloud)
        target_point_cloud_v.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([registered_point_cloud_v, target_point_cloud_v])

        # When the coarse registration give an unsuitable,
        # the correspindence set of the fine registration will be too small.
        # The size of correspindence set is checked here,
        # too small -> registration_algorithm() returns None

    registered_point_cloud = fine_registration(target_point_cloud, registered_point_cloud,
                                               max_correspondence_distance=5,
                                               evaluate_coarse_registraion_min_correspindence=100000)

    if registered_point_cloud is None:
        raise CoarseRegistrationExceptin

    # registered = registration_algorithm(target_point_cloud, registered, max_correspondence_distance=1)
    registered_point_cloud = fine_registration(target_point_cloud, registered_point_cloud,
                                               max_correspondence_distance=0.2)

    if visualization:
        print(time.perf_counter())
        registered_point_cloud_v = copy.deepcopy(registered_point_cloud)
        registered_point_cloud_v.paint_uniform_color([1, 1, 0])
        target_point_cloud_v = copy.deepcopy(target_point_cloud)
        target_point_cloud_v.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([registered_point_cloud_v, target_point_cloud_v])

    return registered_point_cloud


class PointCloudHandler:

    def __init__(self):
        self.idx = 1
        self.num_ac = 0
        self.voxel_size = 1
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=400, origin=([0., 0., 0.]))
        self.xyzrgb_global_comb = o3d.geometry.PointCloud()

        self.location_dir = "/home/wbk-ur2/dual_ws/src/agiprobot_control/scripts/SFB_Demo/models/"
        if not Path(self.location_dir).is_dir():
            Path(self.location_dir).mkdir(parents=True)

        file_list = os.listdir(self.location_dir)
        max_idx = -1
        for file_name in file_list:
            if "scan_" in file_name:
                idx = file_name.split('_')[1]
                idx = int(idx)
                if idx > max_idx:
                    max_idx = idx
        self.location_dir += 'scan_' + str(max_idx) + '/'

        print("Location Folder:", self.location_dir)
        if not Path(self.location_dir).is_dir():
            Path(self.location_dir).mkdir(parents=True)

    def update_combined_pcd(self, xyzrgb_global_cur, rotation_per_capture):

        if not np.asarray(self.xyzrgb_global_comb.points).size:
            # point cloud self.xyzrgb_global_comb is empty

            self.xyzrgb_global_comb += xyzrgb_global_cur

        else:
            print("updating...")

            self.xyzrgb_global_comb = zivid_3d_registration(xyzrgb_global_cur, self.xyzrgb_global_comb,
                                                            rotation_per_capture) + \
                                      xyzrgb_global_cur

        self.xyzrgb_global_comb = self.xyzrgb_global_comb.voxel_down_sample(voxel_size=0.1)

    def crop_pc(self, pc, crop_height):
        points = np.asarray(pc.points)
        pc_cropped = pc.select_by_index(np.where(
            (points[:, 0] > 400) & (points[:, 0] < 1000) & (points[:, 1] > 400) & (points[:, 1] < 1000) & (
                    points[:, 2] > crop_height))[0])

        return pc_cropped

    def preprocess_point_cloud(self, pc):
        radius_normal = self.voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = self.voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pc_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc, o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        return pc, pc_fpfh

    def callback_zivid_listener(self, xyzrgb_global_msg):

        xyzrgb_global_cur = np.asarray(xyzrgb_global_msg.data)

        if not xyzrgb_global_cur.shape[0]:
            rospy.loginfo("Something went wrong, Empty Pointcloud received")

        else:
            xyzrgb_global_cur = np.reshape(xyzrgb_global_cur, (-1, 6))
            self.add_pointcloud(xyzrgb_global_cur)

    def add_pointcloud(self, pointcloud):
        point_cloud_open3d = create_open3d_point_cloud(pointcloud)
        print("Recieved Pointcloud")
        print(point_cloud_open3d)

        point_cloud_open3d, _ = o3d.geometry.PointCloud.remove_statistical_outlier(point_cloud_open3d, 100, 2)
        point_cloud_open3d = point_cloud_open3d.voxel_down_sample(voxel_size=0.1)
        point_cloud_open3d = self.crop_pc(point_cloud_open3d, 60)

        # alignment z axis
        point_cloud_open3d_translated = point_cloud_open3d.translate((-629.84, -637.226, 214.449))

        angle = rospy.get_param("/abs_angle")
        a = np.cos((np.pi / 180) * angle)
        b = np.sin((np.pi / 180) * angle)
        roz = np.array([[a, -b, 0, 0], [b, a, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        point_cloud_open3d_rotated = point_cloud_open3d_translated.transform(roz)

        point_cloud_open3d_retranslated = point_cloud_open3d_rotated.translate((629.84, 637.226, -214.449))

        self.update_combined_pcd(point_cloud_open3d_retranslated, 45)

        finished_updating_pub.publish(("Finished Updating PC_" + str(self.idx).zfill(2)))

        filename_pc_single = self.location_dir + "/pc_part_" + str(self.idx) + ".pcd"
        o3d.io.write_point_cloud(filename=filename_pc_single, pointcloud=point_cloud_open3d_retranslated)

        filename_pc = self.location_dir + "/pc_combined.pcd"
        o3d.io.write_point_cloud(filename=filename_pc, pointcloud=self.xyzrgb_global_comb)
        print("Pointcloud saved")


if __name__ == '__main__':
    rospy.init_node('model_builder')
    pc_handler = PointCloudHandler()
    print("PC_Handler gestartet")
    rospy.Subscriber('PointCloud_Channel_global', Float64MultiArray,
                     pc_handler.callback_zivid_listener)
    print("PointCloud_Channel_global Subscribed")
    finished_updating_pub = rospy.Publisher('PC_Handler_Update', String, queue_size=5)
    rospy.spin()
