import numpy as np
import open3d as o3d
import copy


def point2plane_test(target_point_cloud, source_point_cloud, max_correspondence_distance=10):
    pipreg = o3d.pipelines.registration

    reg = pipreg.registration_icp(source_point_cloud, target_point_cloud,
                                  max_correspondence_distance=max_correspondence_distance,
                                  estimation_method=
                                  pipreg.TransformationEstimationPointToPlane()
                                  # pipreg.TransformationEstimationForGeneralizedICP()
                                  # pipreg.TransformationEstimationForColoredICP()
                                  # pipreg.TransformationEstimationPointToPoint()
                                  )

    print(reg.transformation)
    print(reg)
    print(type(reg.transformation))

    result_point_cloud = copy.deepcopy(source_point_cloud)
    result_point_cloud = result_point_cloud.transform(reg.transformation)

    return result_point_cloud


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

    result_point_cloud = copy.deepcopy(source_point_cloud)
    result_point_cloud = result_point_cloud.transform(reg.transformation)

    return result_point_cloud


def visualization(transformation=None, target_point_cloud_with_background=None, source_point_cloud=None,
                  save_dir='E:/datasets/agiprobot/binlabel/registered_pcd.pcd', save=True):
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
        target_point_cloud_with_background = o3d.io.read_point_cloud('E:/datasets/agiprobot/binlabel/one_view_bin.pcd',
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

    if save:
        o3d.io.write_point_cloud(save_dir, srcDraw + tarDraw, write_ascii=True)

    # tarDraw.paint_uniform_color([0, 1, 1])
    srcDraw.transform(transformation)
    o3d.visualization.draw_geometries([srcDraw, tarDraw])


def translation(target_point_cloud, source_point_cloud):
    target_points = np.array(target_point_cloud.points)
    source_points = np.array(source_point_cloud.points)

    target_points_center = np.sum(target_points, axis=0) / target_points.shape[0]
    source_points_center = np.sum(source_points, axis=0) / source_points.shape[0]

    translated_pionts = source_points - source_points_center + target_points_center + np.array([0, 0, 50])

    translated_piont_cloud = copy.deepcopy(source_point_cloud)
    translated_piont_cloud.points = o3d.utility.Vector3dVector(translated_pionts)

    return translated_piont_cloud


def rotation(source_point_cloud):
    rotated_point_cloud = copy.deepcopy(source_point_cloud)

    euler_angle = rotated_point_cloud.get_rotation_matrix_from_xyz((-np.pi * 30. / 180., np.pi * 3 / 4., 0))
    rotated_point_cloud.rotate(euler_angle)

    return rotated_point_cloud


def coarse_registration(target_point_cloud, source_point_cloud):
    translated_piont_cloud = translation(target_point_cloud, source_point_cloud)
    rotated_point_cloud = rotation(translated_piont_cloud)

    result_piont_cloud = rotated_point_cloud

    return result_piont_cloud


def pipline_point2point(target_point_cloud, source_point_cloud):
    coarse_registered = coarse_registration(target_point_cloud, source_point_cloud)

    '''visualization(save_dir='E:/datasets/agiprobot/binlabel/coarse_registered_pcd.pcd',
                  source_point_cloud=coarse_registered,
                  save=True)'''

    registered = point2point(target_point_cloud, coarse_registered, max_correspondence_distance=10)
    registered = point2point(target_point_cloud, registered, max_correspondence_distance=1)
    registered = point2point(target_point_cloud, registered, max_correspondence_distance=0.1)
    # registered = point2point(target_point_cloud, registered, max_correspondence_distance=0.05)

    visualization(save_dir='E:/datasets/agiprobot/binlabel/registered_pcd.pcd', source_point_cloud=registered,
                  # target_point_cloud_with_background=target_point_cloud,
                  save=True)


def pipline_point2plane_test(target_point_cloud, source_point_cloud):
    '''
    target and source are swapped, since target is full model and computation of its normal is easier.


    :param target_point_cloud:
    :param source_point_cloud:
    :return:
    '''

    coarse_registered = coarse_registration(target_point_cloud, source_point_cloud)

    '''visualization(save_dir='E:/datasets/agiprobot/binlabel/coarse_registered_pcd.pcd',
                  source_point_cloud=coarse_registered,
                  save=True)'''

    radius = 1  # 5  # 1 # 0.5 # 0.1 # 0.01  # max search radius
    max_nn = 30  # max points in the search sphere
    coarse_registered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    '''o3d.visualization.draw_geometries([coarse_registered], window_name="normal estimation",
                                      point_show_normal=True,
                                      width=800,
                                      height=600)'''

    registered = point2plane_test(coarse_registered, target_point_cloud, max_correspondence_distance=10)
    registered = point2plane_test(coarse_registered, registered, max_correspondence_distance=1)
    registered = point2plane_test(coarse_registered, registered, max_correspondence_distance=0.1)
    # registered = point2plane_test(coarse_registered, registered, max_correspondence_distance=0.05)

    visualization(save_dir='E:/datasets/agiprobot/binlabel/registered_pcd.pcd', source_point_cloud=registered,
                  target_point_cloud_with_background=coarse_registered,
                  save=True)


def point2plane(target_point_cloud, source_point_cloud, max_correspondence_distance=10):
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

    print(reg.transformation)
    print(reg)

    rotation_matrix = np.linalg.inv(reg.transformation[0:3, 0:3])
    translation_vector = reg.transformation[0:3, 3] * (-1)
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation_vector
    transformation_matrix[3, 3] = 1.

    print(transformation_matrix)

    result_point_cloud = copy.deepcopy(source_point_cloud)
    result_point_cloud = result_point_cloud.transform(transformation_matrix)

    return result_point_cloud


def pipline_point2plane(target_point_cloud, source_point_cloud):
    coarse_registered = coarse_registration(target_point_cloud, source_point_cloud)

    '''visualization(save_dir='E:/datasets/agiprobot/binlabel/coarse_registered_pcd.pcd',
                  source_point_cloud=coarse_registered,
                  save=True)'''

    '''o3d.visualization.draw_geometries([coarse_registered], window_name="normal estimation",
                                      point_show_normal=True,
                                      width=800,
                                      height=600)'''

    registered = point2plane(target_point_cloud, coarse_registered, max_correspondence_distance=10)
    registered = point2plane(target_point_cloud, registered, max_correspondence_distance=1)
    registered = point2plane(target_point_cloud, registered, max_correspondence_distance=0.1)
    # registered = point2plane(coarse_registered, registered, max_correspondence_distance=0.05)

    visualization(save_dir='E:/datasets/agiprobot/binlabel/registered_pcd.pcd', source_point_cloud=registered,
                  # target_point_cloud_with_background=coarse_registered,
                  save=True)


if __name__ == "__main__":
    target_point_cloud = o3d.io.read_point_cloud('E:/datasets/agiprobot/binlabel/one_view_motor_only.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)

    source_point_cloud = o3d.io.read_point_cloud('E:/datasets/agiprobot/binlabel/full_model.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)

    # pipline_point2point(target_point_cloud, source_point_cloud)

    pipline_point2plane_test(target_point_cloud, source_point_cloud)

    # pipline_point2plane(target_point_cloud, source_point_cloud) # TODO: unsolved error in z-direction
