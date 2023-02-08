import random

import numpy as np
import torch

if __name__ == '__main__':
    pass


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    for b in range(B):
        pc = batch_data[b]
        centroid = torch.mean(pc, dim=0, keepdim=True)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1, keepdim=True)))
        pc = pc / m
        batch_data[b] = pc
    return batch_data


def rotate_per_batch(data, goals, angle_clip=np.pi * 1):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BXNx6 array, original batch of point clouds and point normals
        Return:
          BXNx3 array, rotated batch of point clouds
    """
    if goals != None:
        data = data.float()
        goals = goals.float()
        rotated_data = torch.zeros(data.shape, dtype=torch.float32)
        rotated_data = rotated_data.cuda()

        rotated_goals = torch.zeros(goals.shape, dtype=torch.float32).cuda()
        batch_size = data.shape[0]
        rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=torch.float32).cuda()
        for k in range(data.shape[0]):
            angles = []
            for i in range(3):
                angles.append(random.uniform(-angle_clip, angle_clip))
            angles = np.array(angles)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            R = torch.from_numpy(R).float().cuda()
            rotated_data[k, :, :] = torch.matmul(data[k, :, :], R)
            rotated_goals[k, :, :] == torch.matmul(goals[k, :, :], R)
            rotation_matrix[k, :, :] = R
        return rotated_data, rotated_goals, rotation_matrix
    else:
        data = data.float()
        rotated_data = torch.zeros(data.shape, dtype=torch.float32)
        rotated_data = rotated_data.cuda()

        batch_size = data.shape[0]
        rotation_matrix = torch.zeros((batch_size, 3, 3), dtype=torch.float32).cuda()
        for k in range(data.shape[0]):
            angles = []
            for i in range(3):
                angles.append(random.uniform(-angle_clip, angle_clip))
            angles = np.array(angles)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            R = torch.from_numpy(R).float().cuda()
            rotated_data[k, :, :] = torch.matmul(data[k, :, :], R)
            rotation_matrix[k, :, :] = R
        return rotated_data, rotation_matrix


def base_to_camera(xyz, calc_angle=False):
    '''
    now do the base to camera transform
    '''

    # squeeze the first two dimensions
    xyz_transformed2 = xyz.reshape(-1, 3)  # [N=X*Y, 3]

    # homogeneous transformation
    if calc_angle:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.zeros((xyz_transformed2.shape[0], 1))))  # [N, 4]
    else:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.ones((xyz_transformed2.shape[0], 1))))  # [N, 4]

    cam_to_base_transform_ = np.matrix(cam_to_base_transform)
    base_to_cam_transform = cam_to_base_transform_.I
    xyz_transformed2 = np.matmul(base_to_cam_transform, xyz_transformed2.T).T  # [N, 4]

    return xyz_transformed2[:, :-1].reshape(xyz.shape)  # [X, Y, 3]


def cut_motor(whole_scene):
    x_far = -360
    x_close = -230
    y_far = -910
    y_close = -610
    z_down = 130
    z_up = 300
    Corners = [(x_close, y_far, z_up), (x_close, y_close, z_up), (x_far, y_close, z_up), (x_far, y_far, z_up),
               (x_close, y_far, z_down), (x_close, y_close, z_down), (x_far, y_close, z_down), (x_far, y_far, z_down)]
    # Corners = [(35,880,300), (35,1150,300), (-150,1150,300), (-150,880,300), (35,880,50), (35,1150,50), (-150,1150,50), (-150,880,50)]
    cor_inCam = []
    for corner in Corners:
        cor_inCam_point = base_to_camera(np.array(corner))
        cor_inCam.append(np.squeeze(np.array(cor_inCam_point)))

    panel_1 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[2])
    panel_2 = get_panel(cor_inCam[5], cor_inCam[6], cor_inCam[4])
    panel_3 = get_panel(cor_inCam[0], cor_inCam[3], cor_inCam[4])
    panel_4 = get_panel(cor_inCam[1], cor_inCam[2], cor_inCam[5])
    panel_5 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[4])
    panel_6 = get_panel(cor_inCam[2], cor_inCam[3], cor_inCam[6])
    panel_list = {'panel_up': panel_1, 'panel_bot': panel_2, 'panel_front': panel_3, 'panel_behind': panel_4,
                  'panel_right': panel_5, 'panel_left': panel_6}

    patch_motor = []
    residual_scene = []
    for point in whole_scene:
        point_cor = (point[0], point[1], point[2])
        if set_Boundingbox(panel_list, point_cor):
            patch_motor.append(point)
        else:
            residual_scene.append(point)
    return np.array(patch_motor), np.array(residual_scene)


cam_to_base_transform = [[6.3758686e-02, 9.2318553e-01, -3.7902945e-01, 4.5398907e+01],
                         [9.8811066e-01, -5.1557920e-03, 1.5365793e-01, -7.5876160e+02],
                         [1.3990058e-01, -3.8432005e-01, -9.1253817e-01, 9.6543054e+02],
                         [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]


def get_panel(point_1, point_2, point_3):
    x1 = point_1[0]
    y1 = point_1[1]
    z1 = point_1[2]

    x2 = point_2[0]
    y2 = point_2[1]
    z2 = point_2[2]

    x3 = point_3[0]
    y3 = point_3[1]
    z3 = point_3[2]

    a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
    c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    d = 0 - (a * x1 + b * y1 + c * z1)

    return (a, b, c, d)


def set_Boundingbox(panel_list, point_cor):
    if panel_list['panel_up'][0] * point_cor[0] + panel_list['panel_up'][1] * point_cor[1] + panel_list['panel_up'][2] * \
            point_cor[2] + panel_list['panel_up'][3] <= 0:  # panel 1
        if panel_list['panel_bot'][0] * point_cor[0] + panel_list['panel_bot'][1] * point_cor[1] + \
                panel_list['panel_bot'][2] * point_cor[2] + panel_list['panel_bot'][3] >= 0:  # panel 2
            if panel_list['panel_front'][0] * point_cor[0] + panel_list['panel_front'][1] * point_cor[1] + \
                    panel_list['panel_front'][2] * point_cor[2] + panel_list['panel_front'][3] <= 0:  # panel 3
                if panel_list['panel_behind'][0] * point_cor[0] + panel_list['panel_behind'][1] * point_cor[1] + \
                        panel_list['panel_behind'][2] * point_cor[2] + panel_list['panel_behind'][3] >= 0:  # panel 4
                    if panel_list['panel_right'][0] * point_cor[0] + panel_list['panel_right'][1] * point_cor[1] + \
                            panel_list['panel_right'][2] * point_cor[2] + panel_list['panel_right'][3] >= 0:  # panel 5
                        if panel_list['panel_left'][0] * point_cor[0] + panel_list['panel_left'][1] * point_cor[1] + \
                                panel_list['panel_left'][2] * point_cor[2] + panel_list['panel_left'][
                            3] >= 0:  # panel 6

                            return True
    return False


def camera_to_base(xyz, calc_angle=False):
    '''
    '''
    # squeeze the first two dimensions
    xyz_transformed2 = xyz.reshape(-1, 3)  # [N=X*Y, 3]

    # homogeneous transformation
    if calc_angle:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.zeros((xyz_transformed2.shape[0], 1))))  # [N, 4]
    else:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.ones((xyz_transformed2.shape[0], 1))))  # [N, 4]

    xyz_transformed2 = np.matmul(cam_to_base_transform, xyz_transformed2.T).T  # [N, 4]

    return xyz_transformed2[:, :-1].reshape(xyz.shape)  # [X, Y, 3]