import time

import open3d as o3d

from data_preprocess.registration import _coarse_registration_hard_coding, point2plane, _point2plane_test, point2point
from utilities.data_visualization import visualization_point_cloud


def _running_time():
    target_point_cloud = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/one_view_motor_only.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)

    source_point_cloud = o3d.io.read_point_cloud('E:/datasets/agiprobot/registration/full_model.pcd',
                                                 remove_nan_points=True, remove_infinite_points=True,
                                                 print_progress=True)

    coarse_registered = _coarse_registration_hard_coding(target_point_cloud, source_point_cloud)

    T = time.perf_counter()

    registered = point2plane(target_point_cloud, coarse_registered, max_correspondence_distance=10)
    registered = point2plane(target_point_cloud, registered, max_correspondence_distance=1)
    registered = point2plane(target_point_cloud, registered,
                             max_correspondence_distance=0.1)  # point2plane#point2point

    print(time.perf_counter())

    visualization_point_cloud(save_dir='E:/datasets/agiprobot/binlabel/registered_pcd.pcd',
                              source_point_cloud=registered,
                              # target_point_cloud_with_background=coarse_registered,
                              save=True)


def _pipline_point2plane(target_point_cloud, source_point_cloud):
    coarse_registered = _coarse_registration_hard_coding(target_point_cloud, source_point_cloud)

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

    visualization_point_cloud(save_dir='E:/datasets/agiprobot/binlabel/registered_pcd.pcd',
                              source_point_cloud=registered,
                              # target_point_cloud_with_background=coarse_registered,
                              save=True)


def _pipline_point2plane_test(target_point_cloud, source_point_cloud):
    '''
    target and source are swapped, since target is full model and computation of its normal is easier.


    :param target_point_cloud:
    :param source_point_cloud:
    :return:
    '''

    coarse_registered = _coarse_registration_hard_coding(target_point_cloud, source_point_cloud)

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

    registered = _point2plane_test(coarse_registered, target_point_cloud, max_correspondence_distance=10)
    registered = _point2plane_test(coarse_registered, registered, max_correspondence_distance=1)
    registered = _point2plane_test(coarse_registered, registered, max_correspondence_distance=0.1)
    # registered = point2plane_test(coarse_registered, registered, max_correspondence_distance=0.05)

    # print('Hausdorff\n')
    # print(hausdorff_distance(registered, target_point_cloud))

    visualization_point_cloud(save_dir='E:/datasets/agiprobot/binlabel/registered_pcd.pcd',
                              source_point_cloud=registered,
                              target_point_cloud_with_background=coarse_registered,
                              save=True)


def _pipline_point2point(target_point_cloud, source_point_cloud):
    coarse_registered = _coarse_registration_hard_coding(target_point_cloud, source_point_cloud)

    '''visualization(save_dir='E:/datasets/agiprobot/binlabel/coarse_registered_pcd.pcd',
                  source_point_cloud=coarse_registered,
                  save=True)'''

    registered = point2point(target_point_cloud, coarse_registered, max_correspondence_distance=10)
    registered = point2point(target_point_cloud, registered, max_correspondence_distance=1)
    registered = point2point(target_point_cloud, registered, max_correspondence_distance=0.1)
    # registered = point2point(target_point_cloud, registered, max_correspondence_distance=0.05)

    visualization_point_cloud(save_dir='E:/datasets/agiprobot/binlabel/registered_pcd.pcd',
                              source_point_cloud=registered,
                              # target_point_cloud_with_background=target_point_cloud,
                              save=True)