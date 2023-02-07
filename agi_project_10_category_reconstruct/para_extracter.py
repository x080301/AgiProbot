import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from data_pre_process.dataloader import MotorDataset_patch
import torch.nn as nn
import random
from sklearn.cluster import DBSCAN
import open3d as o3d
import os
from model.model_rotation import PCT_semseg
import csv
import warnings

cam_to_base_transform = [[6.3758686e-02, 9.2318553e-01, -3.7902945e-01, 4.5398907e+01],
                         [9.8811066e-01, -5.1557920e-03, 1.5365793e-01, -7.5876160e+02],
                         [1.3990058e-01, -3.8432005e-01, -9.1253817e-01, 9.6543054e+02],
                         [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]

color_map = {"back_ground": [0, 0, 128],
             "cover": [0, 100, 0],
             "gear_container": [0, 255, 0],
             "charger": [255, 255, 0],
             "bottom": [255, 165, 0],
             "bolts": [255, 0, 0],
             "side_bolts": [255, 0, 255],
             "upgear_a": [224, 255, 255],
             "lowgear_a": [255, 228, 255],
             "gear_b": [230, 230, 255]}


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


def find_geara(seg_motor):
    gearaup = []
    for point in seg_motor:
        if point[3] == 7: gearaup.append(point[0:4])
    gearadown = []
    for point in seg_motor:
        if point[3] == 8: gearadown.append(point[0:4])

    positionaup = np.squeeze(np.mean(gearaup[0:3], axis=1))
    positionadown = np.squeeze(np.mean(gearadown[0:3], axis=1))

    return np.vstack((gearaup, gearadown)), positionaup, positionadown


def find_gearb(seg_motor):
    gearb = []
    for point in seg_motor:
        if point[3] == 9:
            gearb.append(point[0:4])

    positionb = np.squeeze(np.mean(gearb[0:3], axis=1))

    return np.array(gearb), positionb


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


def predict(points):
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--num_heads', type=int, default=4, metavar='num_attention_heads',
                        help='number of attention_heads for self_attention ')
    parser.add_argument('--num_segmentation_type', type=int, default=10, metavar='num_segmentation_type',
                        help='num_segmentation_type)')
    args = parser.parse_args()
    device = torch.device("cuda")
    model = PCT_semseg(args).to(device)
    model = nn.DataParallel(model)
    filename = os.getcwd()
    filename = filename + "/pipeline/merge_model.pth"
    loaded_model = torch.load(filename)
    model.load_state_dict(loaded_model['model_state_dict'])
    TEST_DATASET = MotorDataset_patch(points=points)
    test_loader = DataLoader(TEST_DATASET, num_workers=8, batch_size=16, shuffle=True, drop_last=False)
    num_points_size = points.shape[0]
    result = np.zeros((num_points_size, 4), dtype=float)
    with torch.no_grad():
        model = model.eval()
        cur = 0
        which_type_ret = np.zeros((1))
        for data, data_no_normalize in test_loader:
            data = data.to(device)
            data = normalize_data(data)
            data, GT = rotate_per_batch(data, None)
            data = data.permute(0, 2, 1)
            seg_pred, _, which_type, _, = model(data, 1)
            which_type = which_type.cpu().data.max(1)[1].numpy()
            which_type_ret = np.hstack((which_type_ret, which_type))
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            seg_pred = seg_pred.contiguous().view(-1, 10)  # (batch_size*num_points , num_class)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  # array(batch_size*num_points)
            ##########vis
            points = data_no_normalize.view(-1, 3).cpu().data.numpy()
            pred_choice_ = np.reshape(pred_choice, (-1, 1))
            points = np.hstack((points, pred_choice_))
            # vis(points)
            if cur == 0:
                cur = 1
                result = points
            else:
                result = np.vstack((result, points))
            count = np.bincount(which_type_ret.astype(int))
            type = np.argmax(count)
        return result, type


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


def open3d_save_pcd(pc, filename):
    sampled = np.asarray(pc)
    PointCloud_koordinate = sampled[:, 0:3]

    # visuell the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True)


def find_covers(seg_motor):
    bottom = []
    for point in seg_motor:
        if point[3] == 1: bottom.append(point[0:3])
    bottom = np.array(bottom)
    if bottom.shape[0] < 1000:
        return -1, None, None
    filename = os.getcwd()
    filename = filename + "/data/cover.pcd"
    open3d_save_pcd(bottom, filename)
    pcd = o3d.io.read_point_cloud(filename)
    downpcd = pcd.voxel_down_sample(voxel_size=0.002)  # 下采样滤波，体素边长为0.002m
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))  # 计算法线，只考虑邻域内的20个点
    nor = downpcd.normals
    points = downpcd.points
    normal = []
    for ele in nor:
        normal.append(ele)
    normal = np.array(normal)
    model = DBSCAN(eps=0.02, min_samples=100)
    yhat = model.fit_predict(normal)  # genalize label based on index
    clusters = np.unique(yhat)
    noise = []
    clusters_new = []
    bottom_to_judge = 1
    for i in clusters:
        noise.append(i) if np.sum(i == yhat) < 2000 or i == -1 else clusters_new.append(i)
    for clu in clusters_new:
        row_ix = np.where(yhat == clu)
        normal = np.squeeze(np.mean(normal[row_ix, :3], axis=1))
        normal = np.around(normal, 5)
        bottom_to_judge = np.array(points)[row_ix, :3]
        bottom_to_judge = np.squeeze(bottom_to_judge)
        break
    return 1, bottom_to_judge, normal


def find_bolts(seg_motor, eps, min_points):
    bolts = []
    for point in seg_motor:
        if point[3] == 6: bolts.append(point[0:3])
    bolts = np.asarray(bolts)
    model = DBSCAN(eps=eps, min_samples=min_points)
    yhat = model.fit_predict(bolts)  # genalize label based on index
    clusters = np.unique(yhat)
    noise = []
    clusters_new = []
    positions = []
    for i in clusters:
        noise.append(i) if np.sum(i == yhat) < 50 or i == -1 else clusters_new.append(i)
    flag = 0
    bolts__ = 1
    for clu in clusters_new:
        row_ix = np.where(yhat == clu)
        if flag == 0:
            bolts__ = np.squeeze(np.array(bolts[row_ix, :3]))
            flag = 1
        else:
            inter = np.squeeze(np.array(bolts[row_ix, :3]))
            bolts__ = np.concatenate((bolts__, inter), axis=0)
        np.set_printoptions(precision=2)
        position = np.squeeze(np.mean(bolts[row_ix, :3], axis=1))
        positions.append(position)
    positions = np.array(positions)
    indexs = np.argsort(positions[:, 1])
    positions = positions[indexs, :]

    return positions, len(clusters_new), bolts__


# interface functions for users
class ParaExtracter:
    def __init__(self):
        pass

    def load(self, point_cloud_input_file_name):  # find_action_load

        self.filename_ = point_cloud_input_file_name.split('/')[-1]
        # read point cloud data
        pcd = o3d.io.read_point_cloud(point_cloud_input_file_name)
        colors = np.asarray(pcd.colors)
        points = np.asarray(pcd.points)
        cloud = np.concatenate([points, colors], axis=-1)

        num_points = len(cloud)

        points_to_model = []
        for i in range(num_points):
            dp = cloud[i]

            r = int(cloud[i][3] * 255)
            g = int(cloud[i][4] * 255)
            b = int(cloud[i][5] * 255)
            points_to_model.append([dp[0], dp[1], dp[2], r, g, b])

        return points_to_model, num_points

    def predict(self, points_to_model):  # find_pushButton
        # ******************************
        # cut_cuboid
        # ******************************
        points_to_model = np.array(points_to_model)
        motor_scene, residual_scene = cut_motor(points_to_model)

        current_points_size = motor_scene.shape[0]
        if current_points_size % 2048 != 0:
            num_add_points = 2048 - (current_points_size % 2048)
            choice = np.random.choice(current_points_size, num_add_points, replace=True)
            add_points = motor_scene[choice, :]
            motor_points = np.vstack((motor_scene, add_points))
            np.random.shuffle(motor_points)
        else:
            motor_points = motor_scene
            np.random.shuffle(motor_points)

        # ******************************
        # find_predict
        # ******************************
        motor_points_forecast, self.type = predict(motor_points[:, 0:3])

        self.transfer_to_borot_coordinate()

        return motor_points_forecast, self.type

    def transfer_to_borot_coordinate(self, motor_points_forecast):

        motor_points_forecast_in_robot = np.random.rand(motor_points_forecast.shape[0], 4)
        motor_points_forecast_in_robot[:, 0:3] = np.array(camera_to_base(motor_points_forecast[:, 0:3]))
        motor_points_forecast_in_robot[:, 3] = np.array(motor_points_forecast[:, 3])

        self.cover_existence, self.covers, self.normal = find_covers(motor_points_forecast_in_robot)

        return motor_points_forecast_in_robot

    def save(self, motor_points_forecast):  # find_action_save

        motor_points_forecast_in_robot = self.transfer_to_borot_coordinate(motor_points_forecast)
        sampled = np.asarray(motor_points_forecast_in_robot)
        PointCloud_koordinate = sampled[:, 0:3]
        label = sampled[:, 3]
        labels = np.asarray(label)
        colors = []
        for i in range(labels.shape[0]):
            dp = labels[i]
            if dp == 0:
                r = color_map["back_ground"][0]
                g = color_map["back_ground"][1]
                b = color_map["back_ground"][2]
                colors.append([r, g, b])
            elif dp == 1:
                r = color_map["cover"][0]
                g = color_map["cover"][1]
                b = color_map["cover"][2]
                colors.append([r, g, b])
            elif dp == 2:
                r = color_map["gear_container"][0]
                g = color_map["gear_container"][1]
                b = color_map["gear_container"][2]
                colors.append([r, g, b])
            elif dp == 3:
                r = color_map["charger"][0]
                g = color_map["charger"][1]
                b = color_map["charger"][2]
                colors.append([r, g, b])
            elif dp == 4:
                r = color_map["bottom"][0]
                g = color_map["bottom"][1]
                b = color_map["bottom"][2]
                colors.append([r, g, b])
            elif dp == 5:
                r = color_map["side_bolts"][0]
                g = color_map["side_bolts"][1]
                b = color_map["side_bolts"][2]
                colors.append([r, g, b])
            elif dp == 6:
                r = color_map["bolts"][0]
                g = color_map["bolts"][1]
                b = color_map["bolts"][2]
                colors.append([r, g, b])
            elif dp == 8:
                r = color_map["upgear_a"][0]
                g = color_map["upgear_a"][1]
                b = color_map["upgear_a"][2]
                colors.append([r, g, b])
            elif dp == 7:
                r = color_map["lowgear_a"][0]
                g = color_map["lowgear_a"][1]
                b = color_map["lowgear_a"][2]
                colors.append([r, g, b])
            else:
                r = color_map["gear_b"][0]
                g = color_map["gear_b"][1]
                b = color_map["gear_b"][2]
                colors.append([r, g, b])
        colors = np.array(colors)
        colors = colors / 255
        # print(colors)

        # visuell the point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([point_cloud])
        filename = os.getcwd()
        if not os.path.exists(
                'predicted_result'):  # initial the file, if not exiting (os.path.exists() is pointed at ralative position and current cwd)
            os.makedirs('predicted_result')
        FileName = filename + '/' + 'predicted_result'
        if not os.path.exists(
                FileName):  # initial the file, if not exiting (os.path.exists() is pointed at ralative position and current cwd)
            os.makedirs(FileName)
        FileName__ = FileName + '/' + self.filename_.split('.')[0] + "_segmentation"
        o3d.io.write_point_cloud(FileName__ + ".pcd", point_cloud)

        # self.positions_bolts,self.num_bolts,
        # self.normal
        if self.cover_existence > 0:
            csv_path = FileName__ + '.csv'
            with open(csv_path, 'a+', newline='') as f:
                csv_writer = csv.writer(f)
                head = ["     ", "x", "y", "z", "Rx", "Ry", "Rz"]
                csv_writer.writerow(head)
                for i in range(self.num_bolts):
                    head = ["screw_" + str(i + 1), str(self.positions_bolts[i][0]), str(self.positions_bolts[i][1]),
                            str(self.positions_bolts[i][2]),
                            str(self.normal[0]), str(self.normal[1]), str(self.normal[2])]
                    csv_writer.writerow(head)
        else:
            csv_path = FileName__ + '.csv'
            with open(csv_path, 'a+', newline='') as f:
                csv_writer = csv.writer(f)
                head = ["     ", "x", "y", "z"]
                csv_writer.writerow(head)
                if self.type <= 2:
                    head = ["TypeA_upper_gear", str(self.posgearaup[0]), str(self.posgearaup[1]),
                            str(self.posgearaup[2])]
                    csv_writer.writerow(head)
                    head = ["TypeA_lower_gear", str(self.posgearadown[0]), str(self.posgearadown[1]),
                            str(self.posgearadown[2])]
                    csv_writer.writerow(head)
                else:
                    head = ["TypeB_gear", str(self.posgearb[0]), str(self.posgearb[1]), str(self.posgearb[2])]
                    csv_writer.writerow(head)

    def filter_bolts(self, motor_points_forecast_in_robot):  # find_action_bolts_position
        if self.cover_existence <= 0:
            warnings.warn("Cover has been removed\nno cover screw found", UserWarning)

            return
        self.positions_bolts, self.num_bolts, self.bolts = find_bolts(motor_points_forecast_in_robot, eps=2.5,
                                                                      min_points=50)

        return self.positions_bolts, self.num_bolts, self.bolts

    def display_gear(self, motor_points_forecast_in_robot):  # find_actionGear_position
        if self.cover_existence > 0:
            warnings.warn("Cover has been removed\nno cover screw found", UserWarning)
            return

        gearpositions = []
        if self.type <= 2:
            self.gear, self.posgearaup, self.posgearadown = find_geara(
                seg_motor=motor_points_forecast_in_robot)
            gearpositions.append(self.posgearaup, self.posgearadown)
        else:
            self.gear, self.posgearb = find_gearb(seg_motor=motor_points_forecast_in_robot)
            gearpositions.append(self.posgearb)

        return self.gear, gearpositions


if __name__ == "__main__":
    pass
