import numpy as np
import open3d as o3d
import copy


def _point2plane_test(target_point_cloud, source_point_cloud, max_correspondence_distance=10):
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


def _translation(target_point_cloud, source_point_cloud):
    translation_vector = target_point_cloud.get_center() - source_point_cloud.get_center()

    source_point_cloud.translate(translation_vector, relative=True)

    return source_point_cloud


def _rotation(source_point_cloud):
    rotated_point_cloud = copy.deepcopy(source_point_cloud)

    euler_angle = rotated_point_cloud.get_rotation_matrix_from_xyz((-np.pi * 30. / 180., np.pi * 3 / 4., 0))
    rotated_point_cloud.rotate(euler_angle)

    return rotated_point_cloud


def _coarse_registration_hard_coding(target_point_cloud, source_point_cloud):
    translated_piont_cloud = _translation(target_point_cloud, source_point_cloud)
    rotated_point_cloud = _rotation(translated_piont_cloud)

    result_piont_cloud = rotated_point_cloud

    return result_piont_cloud


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
    reg = pipreg.registration_icp(target_point_cloud, source_point_cloud,  # (target_point_cloud, source_point_cloud,
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

        return None, None

    else:

        result_point_cloud = copy.deepcopy(source_point_cloud)

        transform_matrix = np.linalg.inv(np.asarray(reg.transformation))
        result_point_cloud.transform(transform_matrix)
        # result_point_cloud = result_point_cloud.transform(transformation_matrix)
        '''
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
        '''
        return result_point_cloud, transform_matrix


def get_motor_only_pcd(target_point_cloud):
    colors = np.array(target_point_cloud.colors)
    points = np.array(target_point_cloud.points)

    target_index = np.argwhere(np.sum(colors, axis=1) > 1)
    motor_only = points[target_index, :].reshape((-1, 3))

    motor_only_point_cloud = o3d.geometry.PointCloud()
    motor_only_point_cloud.points = o3d.utility.Vector3dVector(motor_only)

    return motor_only_point_cloud


class CoarseRegistrationExceptin(Exception):
    "this is user's Exception for unsuccessful coarse registration "

    def __init__(self):
        pass

    def __str__(self):
        print("coarse registration failed")


if __name__ == "__main__":
    from pipeline import get_transform_matrix

    registered_point_cloud, transform_matrix = get_transform_matrix(
        r'E:\datasets\agiprobot\pipeline_demo\2_t_colored.pcd',
        r'E:\datasets\agiprobot\pipeline_demo\2_full_model.pcd', visualization=True)
    print(transform_matrix)
