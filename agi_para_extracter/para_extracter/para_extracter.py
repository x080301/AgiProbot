import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from para_extracter.data_postprocess.data_postprecess import find_geara, find_gearb, find_bolts
from para_extracter.data_preprocess.data_preprocess import normalize_data, rotate_per_batch, cut_motor, camera_to_base
from para_extracter.data_preprocess.dataloader import MotorDataset_patch
import torch.nn as nn
from sklearn.cluster import DBSCAN
import open3d as o3d
from para_extracter.model.model_rotation import PCT_semseg
import csv
import warnings

import os


def open3d_save_pcd(pc, filename):
    sampled = np.asarray(pc)
    PointCloud_koordinate = sampled[:, 0:3]

    # visuell the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True)


def find_covers(seg_motor, cover_file_dir=None):
    bottom = []
    for point in seg_motor:
        if point[3] == 1: bottom.append(point[0:3])
    bottom = np.array(bottom)
    if bottom.shape[0] < 1000:
        return -1, None, None

    if cover_file_dir is None:
        cover_file_dir = os.path.dirname(__file__) + '/cover.pcd'

    open3d_save_pcd(bottom, cover_file_dir)
    pcd = o3d.io.read_point_cloud(cover_file_dir)
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


# interface functions for users
class ParaExtracter:
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

    def __init__(self):
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
        self.args = parser.parse_args()

        self.device = torch.device("cuda")

        self.model = PCT_semseg(self.args).to(self.device)
        self.model = nn.DataParallel(self.model)

    def load_pcd_data(self, point_cloud_input_file_name):  # find_action_load
        """
        load_pcd_data(point_cloud_input_file_name)
        Function to load pointcloud data from file

        Args:
            point_cloud_input_file_name (str): Path to pcd file.

        Returns:
            read point cloud and number of points, which are raised for debugging only.
        """

        # read point cloud data
        point_cloud = o3d.io.read_point_cloud(point_cloud_input_file_name,
                                              remove_nan_points=True, remove_infinite_points=True, print_progress=True)
        colors = np.asarray(point_cloud.colors)
        points = np.asarray(point_cloud.points)
        point_cloud = np.concatenate([points, colors], axis=-1)

        num_points = len(point_cloud)

        point_cloud[:, 3] *= 225
        point_cloud[:, 4] *= 225
        point_cloud[:, 5] *= 225

        self.point_cloud = point_cloud

        return self.point_cloud, num_points

    def load_model(self, model_file_dir='not defined'):
        """
        load_model(model_file_dir='not defined')
        Function to load trained model from file

        Args:
            model_file_dir (str, optional, default='not defined'): Path to trained model. When not specified or set as ``not defined``, the model is read from file './merge_model.pth'.

        Returns:
            None
        """

        if model_file_dir == 'not defined':
            model_file_dir = os.path.dirname(__file__) + '/merge_model.pth'

        model_file_dir = model_file_dir  # + "/pipeline/merge_model.pth"
        loaded_model = torch.load(model_file_dir)

        self.model.load_state_dict(loaded_model['model_state_dict'])

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

        motor_points = motor_points[:, 0:3]

        TEST_DATASET = MotorDataset_patch(points=motor_points)
        test_loader = DataLoader(TEST_DATASET, num_workers=8, batch_size=16, shuffle=True, drop_last=False)
        '''num_points_size = motor_points.shape[0]
        motor_points_forecast = np.zeros((num_points_size, 4), dtype=float)'''
        with torch.no_grad():
            self.model = self.model.eval()
            cur = 0
            which_type_ret = np.zeros((1))
            for data, data_no_normalize in test_loader:
                data = data.to(self.device)
                data = normalize_data(data)  # (B,N,3)
                data, GT = rotate_per_batch(data, None)
                data = data.permute(0, 2, 1)  # (B,3,N)
                seg_pred, _, which_type, _, = self.model(data, 1)
                which_type = which_type.cpu().data.max(1)[1].numpy()
                which_type_ret = np.hstack((which_type_ret, which_type))
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                seg_pred = seg_pred.contiguous().view(-1, 10)  # (batch_size*num_points , num_class)
                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  # array(batch_size*num_points)
                ##########vis
                motor_points = data_no_normalize.view(-1, 3).cpu().data.numpy()
                pred_choice_ = np.reshape(pred_choice, (-1, 1))
                motor_points = np.hstack((motor_points, pred_choice_))
                # vis(points)
                if cur == 0:
                    cur = 1
                    motor_points_forecast = motor_points
                else:
                    motor_points_forecast = np.vstack((motor_points_forecast, motor_points))
                count = np.bincount(which_type_ret.astype(int))
                self.type = np.argmax(count)

        return motor_points_forecast, self.type

    def run(self):
        """
        run()
        Function to run the predition of the model. Relevant data are calculated and stored automatically.

        Args:
            None
        Returns:
            None
        """
        self.segementation_prediction, self.classification_prediction = self.predict(self.point_cloud)

        self.segementation_prediction_in_robot = self.transfer_to_robot_coordinate(self.segementation_prediction)
        self._cover_existence, self.covers, self.normal = find_covers(self.segementation_prediction_in_robot)

    def get_segmentation_prediction(self):
        """
        get_segmentation_prediction()
        Function to get the segmentation results of the model.
        return segementation prediction, with type numpy.ndarray and with shape (Num_points, 4)
        Dimension1 of the return: 4 = [x, y, z, segemention_predition]

        Args:
            None
        Returns:
            numpy.ndarray with shape (Num_points, 4)
        """

        return self.segementation_prediction

    def get_classification_prediction(self):
        """
        get_classification_prediction()
        Function to get the classification results of the model.

        Args:
            None
        Returns:
            classification (numpy.int64)
        """
        return self.classification_prediction

    def transfer_to_robot_coordinate(self, motor_points_forecast):

        motor_points_forecast_in_robot = np.random.rand(motor_points_forecast.shape[0], 4)
        motor_points_forecast_in_robot[:, 0:3] = np.array(camera_to_base(motor_points_forecast[:, 0:3]))
        motor_points_forecast_in_robot[:, 3] = np.array(motor_points_forecast[:, 3])

        # self.cover_existence, self.covers, self.normal = find_covers(motor_points_forecast_in_robot)

        return motor_points_forecast_in_robot

    def save(self, save_dir, motor_points_forecast):  # find_action_save

        motor_points_forecast_in_robot = self.transfer_to_robot_coordinate(motor_points_forecast)
        sampled = np.asarray(motor_points_forecast_in_robot)
        PointCloud_koordinate = sampled[:, 0:3]
        label = sampled[:, 3]
        labels = np.asarray(label)
        colors = []
        for i in range(labels.shape[0]):
            dp = labels[i]
            if dp == 0:
                r = self.color_map["back_ground"][0]
                g = self.color_map["back_ground"][1]
                b = self.color_map["back_ground"][2]
                colors.append([r, g, b])
            elif dp == 1:
                r = self.color_map["cover"][0]
                g = self.color_map["cover"][1]
                b = self.color_map["cover"][2]
                colors.append([r, g, b])
            elif dp == 2:
                r = self.color_map["gear_container"][0]
                g = self.color_map["gear_container"][1]
                b = self.color_map["gear_container"][2]
                colors.append([r, g, b])
            elif dp == 3:
                r = self.color_map["charger"][0]
                g = self.color_map["charger"][1]
                b = self.color_map["charger"][2]
                colors.append([r, g, b])
            elif dp == 4:
                r = self.color_map["bottom"][0]
                g = self.color_map["bottom"][1]
                b = self.color_map["bottom"][2]
                colors.append([r, g, b])
            elif dp == 5:
                r = self.color_map["side_bolts"][0]
                g = self.color_map["side_bolts"][1]
                b = self.color_map["side_bolts"][2]
                colors.append([r, g, b])
            elif dp == 6:
                r = self.color_map["bolts"][0]
                g = self.color_map["bolts"][1]
                b = self.color_map["bolts"][2]
                colors.append([r, g, b])
            elif dp == 8:
                r = self.color_map["upgear_a"][0]
                g = self.color_map["upgear_a"][1]
                b = self.color_map["upgear_a"][2]
                colors.append([r, g, b])
            elif dp == 7:
                r = self.color_map["lowgear_a"][0]
                g = self.color_map["lowgear_a"][1]
                b = self.color_map["lowgear_a"][2]
                colors.append([r, g, b])
            else:
                r = self.color_map["gear_b"][0]
                g = self.color_map["gear_b"][1]
                b = self.color_map["gear_b"][2]
                colors.append([r, g, b])
        colors = np.array(colors)
        colors = colors / 255
        # print(colors)

        # visuell the point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([point_cloud])
        if not os.path.exists(
                'predicted_result'):  # initial the file, if not exiting (os.path.exists() is pointed at ralative position and current cwd)
            os.makedirs('predicted_result')
        filename = save_dir + '/predicted_result/'
        if not os.path.exists(
                filename):  # initial the file, if not exiting (os.path.exists() is pointed at ralative position and current cwd)
            os.makedirs(filename)
        filename = filename + self.filename_.split('.')[0] + "_segmentation"
        o3d.io.write_point_cloud(filename + ".pcd", point_cloud)

        # self.positions_bolts,self.num_bolts,
        # self.normal
        if self._cover_existence > 0:
            csv_path = filename + '.csv'
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
            csv_path = filename + '.csv'
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

    def find_screws(self):  # find_action_bolts_position
        """
        find_screws()
        Function to get information of the cover crews.
        return: (None when no screws is found)
                bolt_positions          # numpy.ndarray with shape (cover_crew_num,3). The dimension1 is center of the crew [x,y,z]
                cover_screw_normal      # numpy.ndarray with shape (cover_crew_num,3). The dimension1 is normal of the crew
                bolt_num                # int, number of cover crews
                bolt_piont_clouds       # numpy.ndarray with shape (points_num,3).

        Args:
            None
        Returns:
            bolt_positions, cover_screw_normal, bolt_num, bolt_piont_clouds
        """
        if self._cover_existence <= 0:
            warnings.warn("Cover has been removed\nno cover screw found", UserWarning)
            return None, None, None, None
        self.positions_bolts, self.num_bolts, bolts = find_bolts(self.segementation_prediction_in_robot, eps=2.5,
                                                                 min_points=50)

        normal_cover_screws = self.normal

        return self.positions_bolts, normal_cover_screws, self.num_bolts, bolts

    def if_cover_existence(self):
        if self._cover_existence > 0:
            return True
        else:
            return False

    def find_gears(self):  # find_actionGear_position
        """
        find_gears()
        Function to get information of the gears.
        return:
                gear_piont_clouds          # numpy.ndarray with shape (points_num,3). return None when no gear is found
                gearpositions              # numpy.ndarray with shape (gear_num,3). The dimension1 is center of the gears [x,y,z]. return None when no gear is found

        Args:
            None
        Returns:
            gear_piont_clouds, gearpositions
        """
        if self._cover_existence > 0:
            warnings.warn("\nCover has not been removed\nno gear found", UserWarning)
            return None, None

        gearpositions = []
        if self.type <= 2:
            gear, self.posgearaup, self.posgearadown = find_geara(
                seg_motor=self.segementation_prediction_in_robot)
            gearpositions.append(self.posgearaup)
            gearpositions.append(self.posgearadown)
        else:
            gear, self.posgearb = find_gearb(seg_motor=self.segementation_prediction_in_robot)
            gearpositions.append(self.posgearb)

        return gear, gearpositions


if __name__ == "__main__":
    pass
