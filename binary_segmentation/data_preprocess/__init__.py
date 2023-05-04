
from data_preprocess import alignment
point_cloud_alignment = alignment.pointcloud_alignment

from data_preprocess import registration

registration = registration.registration
'''

    :param target_point_cloud: 
    :param source_point_cloud: 
    :param algorithm: (string, optimal) 'point2plane_multi_step' or 'point2point_multi_step' 
    :param visualization: 
    :return: registered point cloud
'''
