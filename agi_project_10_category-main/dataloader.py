"""
@Author: bixuelei
@Contact: bxueleibi@gmial.com
@File: data_loader.py
@Time: 2022/1/16 3:49 PM
"""

import os
import numpy as np
import random
from numpy.random import choice
from tqdm import tqdm  # used to display the circulation position, to see where the code is running at
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
# from display import Visuell_PointCloud,Visuell,Visuell__
from util import index_points
import torch
from kmeans_pytorch import kmeans


####### normalize the point cloud############
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    max_distance = np.sqrt(np.max(np.sum(pc ** 2, axis=1)))
    pc = pc / max_distance
    return pc


def Get_ObjectID(x):  # get all kinds of ObjectID from numpy file

    dic = []
    for i in range(x.shape[0]):
        if x[i][6] not in dic:
            dic.append(x[i][6])

    return dic


def densify_blots(patch_motor):
    add = []
    for i in range(len(patch_motor)):
        if (patch_motor[i][6] == 6 or patch_motor[i][6] == 5):
            add.append(patch_motor[i])
    add = np.array(add)
    twonn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(add[:, 0:3])
    _, indices = twonn.kneighbors(add[:, 0:3])
    inter = []
    for i in range(indices.shape[0]):
        interpolation = np.zeros(7)
        interpolation[3:7] = add[i][3:7]
        # if the bolt points are closest to eachonter
        if (indices[indices[i][1]][1] == i):
            interpolation[0:3] = add[i][0:3] + (add[indices[i][1]][0:3] - add[i][0:3]) / 3
            inter.append(interpolation)
        else:
            interpolation[0:3] = add[i][0:3] + (add[indices[i][1]][0:3] - add[i][0:3]) / 2
            inter.append(interpolation)
    patch_motor = np.concatenate((patch_motor, inter), axis=0)
    return patch_motor


def knn(x, k):
    """
    Input:
        points: input points data, [B, N, C]
    Return:
        idx: sample index data, [B, N, K]
    """
    # x=x.permute(0,2,1)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def find_goals_kmeans(points__, target__):  # [bs,n_points,C] [bs,n_points]
    # target__=target__.astype(np.float32)
    # points = torch.from_numpy(points).cuda()
    # target= torch.from_numpy(target).cuda()
    goals_ = [8, 8, 8, 8, 8, 8]
    bs = points__.shape[0]
    mask = torch.ones((bs, 48))
    # # device=points.device
    # cover_bolts=torch.zeros(bs,goals_[6],3)
    bolts = torch.zeros(bs, goals_[5], 3)
    bottoms = torch.zeros(bs, goals_[4], 3)
    chargers = torch.zeros(bs, goals_[2], 3)
    gearcontainers = torch.zeros(bs, goals_[3], 3)
    covers = torch.zeros(bs, goals_[1], 3)
    clampingsystem = torch.zeros(bs, goals_[0], 3)
    for i in range(bs):
        target = torch.from_numpy(target__[i])
        points = torch.from_numpy(points__[i])
        points = points[:, 0:3]
        index_clampingsystem = target == 0
        points1 = points[index_clampingsystem, :]
        _, c1 = kmeans(X=points1, num_clusters=goals_[0], distance='euclidean', device=torch.device('cuda'))
        #         c1=c1.cuda()
        dis1 = square_distance(points1.unsqueeze(0).float(), c1.unsqueeze(0).float())
        index1 = dis1.squeeze(0).min(dim=0, keepdim=False)[1]
        added1 = index_points(points1.unsqueeze(0), index1.unsqueeze(0))
        #######################
        # Visuell(points1,c1)
        # Visuell__(points1.squeeze(0),index1)
        #########################
        _, sorted_index_clamping = added1.squeeze(0).sort(dim=0)
        sorted_index_clamping_x = sorted_index_clamping[:, 0]
        goals_clamping = added1[:, sorted_index_clamping_x, :]
        clampingsystem[i, :, :] = goals_clamping[0, :, :]

        index_cover = target == 1
        points2 = points[index_cover, :]
        _, c2 = kmeans(X=points2, num_clusters=goals_[1], distance='euclidean', device=torch.device('cuda'))
        # c2=c2.cuda()
        dis2 = square_distance(points2.unsqueeze(0).float(), c2.unsqueeze(0).float())
        index2 = dis2.squeeze(0).min(dim=0, keepdim=False)[1]
        added2 = index_points(points2.unsqueeze(0), index2.unsqueeze(0))
        #######################
        # Visuell(points2,c2)
        # Visuell__(points2.squeeze(0),index2)
        ###########################
        _, sorted_index_cover = added2.squeeze(0).sort(dim=0)
        sorted_index_cover_x = sorted_index_cover[:, 0]
        goals_cover = added2[:, sorted_index_cover_x, :]
        covers[i, :, :] = goals_cover[0, :, :]

        index_gearcontainer = target == 2
        if torch.sum(index_gearcontainer) >= 8:
            points3 = points[index_gearcontainer, :]
            _, c3 = kmeans(X=points3, num_clusters=goals_[2], distance='euclidean', device=torch.device('cuda'))
            # c3=c3.cuda()
            dis3 = square_distance(points3.unsqueeze(0).float(), c3.unsqueeze(0).float())
            index3 = dis3.squeeze(0).min(dim=0, keepdim=False)[1]
            added3 = index_points(points3.unsqueeze(0), index3.unsqueeze(0))
            #######################
            # Visuell(points3,c3)
            # Visuell__(points3.squeeze(0),index3)
            ###########################
            _, sorted_index_gear = added3.squeeze(0).sort(dim=0)
            sorted_index_gear_x = sorted_index_gear[:, 0]
            goals_gear = added3[:, sorted_index_gear_x, :]
            gearcontainers[i, :, :] = goals_gear[0, :, :]
        else:
            gearcontainers[i, :, :] = torch.zeros((goals_[2], 3))
            mask[i][16:24] = 0

        index_charger = target == 3
        if torch.sum(index_charger) >= 8:
            points4 = points[index_charger, :]
            _, c4 = kmeans(X=points4, num_clusters=goals_[3], distance='euclidean', device=torch.device('cuda'))
            dis4 = square_distance(points4.unsqueeze(0).float(), c4.unsqueeze(0).float())
            index4 = dis4.squeeze(0).min(dim=0, keepdim=False)[1]
            added4 = index_points(points4.unsqueeze(0), index4.unsqueeze(0))
            #######################
            # Visuell(points4,c4)
            # Visuell__(points4.squeeze(0),index4)
            ###########################
            _, sorted_index_charger = added4.squeeze(0).sort(dim=0)
            sorted_index_charger_x = sorted_index_charger[:, 0]
            goals_charger = added4[:, sorted_index_charger_x, :]
            chargers[i, :, :] = goals_charger[0, :, :]
        else:
            chargers[i, :, :] = torch.zeros((goals_[3], 3))
            mask[i][24:32] = 0

        index_bottom = target == 4
        points5 = points[index_bottom, :]
        _, c5 = kmeans(X=points5, num_clusters=goals_[4], distance='euclidean', device=torch.device('cuda'))
        dis5 = square_distance(points5.unsqueeze(0).float(), c5.unsqueeze(0).float())
        index5 = dis5.squeeze(0).min(dim=0, keepdim=False)[1]
        added5 = index_points(points5.unsqueeze(0), index5.unsqueeze(0))
        #######################
        # Visuell(points5,c5)
        # Visuell__(points5.squeeze(0),index5)
        ###########################
        _, sorted_index_bottom = added5.squeeze(0).sort(dim=0)
        sorted_index_bottom_x = sorted_index_bottom[:, 0]
        goals_bottom = added5[:, sorted_index_bottom_x, :]
        bottoms[i, :, :] = goals_bottom[0, :, :]

        index_bolts = target == 5
        if torch.sum(index_bolts) >= 8:
            points6 = points[index_bolts, :]
            _, c6 = kmeans(X=points6, num_clusters=goals_[5], distance='euclidean', device=torch.device('cpu'))
            # c6=c6.cuda()
            dis6 = square_distance(points6.unsqueeze(0).float(), c6.unsqueeze(0).float())
            index6 = dis6.squeeze(0).min(dim=0, keepdim=False)[1]
            added6 = index_points(points6.unsqueeze(0), index6.unsqueeze(0))
            ########################
            # Visuell(points6,c6)
            # Visuell__(points6.squeeze(0),index6)
            ############################
            _, sorted_index_bolt = added6.squeeze(0).sort(dim=0)
            sorted_index_bolt_x = sorted_index_bolt[:, 0]
            goals_bolts = added6[:, sorted_index_bolt_x, :]
            bolts[i, :, :] = goals_bolts[0, :, :]
        else:
            bolts[i, :, :] = torch.ones((goals_[5], 3))
            mask[i][40:48] = 0

    goals = torch.cat((clampingsystem, covers, gearcontainers, chargers, bottoms, bolts), dim=1)
    return goals.numpy(), mask.numpy()


bolt_weights = 1


class MotorDataset(Dataset):
    def __init__(self, split='train', data_root='directory to training data', num_class=6, num_points=4096,
                 bolt_weight=1, test_area='Validation', block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_points = num_points
        self.num_class = num_class
        self.block_size = block_size
        self.transform = transform
        motor_list = sorted(os.listdir(data_root))  # list all subdirectory
        if split == 'train':  # load training files or validation files
            motor_positions = [motor for motor in motor_list if '{}'.format(test_area) not in motor]
        else:
            motor_positions = [motor for motor in motor_list if '{}'.format(test_area) in motor]

        ######################load the np file###################################################

        self.motors_points = []  # initial object_motor_points and object_motor_lables
        self.motors_labels = []
        self.type_for_each_motor = []
        self.cover_exitence_for_each_motor = []
        num_points_eachmotor = []  # initial a list to count the num of points for each motor
        label_num_eachtype = np.zeros(self.num_class)  # initial a array to count how much points is there for each type
        for motor_position in tqdm(motor_positions, total=len(motor_positions)):
            motor_directory = os.path.join(data_root, motor_position)
            motor_data = np.load(motor_directory)
            # motor_data=densify_blots(motor_data)
            if 'A0' in motor_position:
                class_motor = 0
            elif 'A1' in motor_position:
                class_motor = 1
            elif 'A2' in motor_position:
                class_motor = 2
            elif 'B0' in motor_position:
                class_motor = 3
            else:
                class_motor = 4

            if 'gear' in motor_position:
                cover_exitence_motor = 0
            else:
                cover_exitence_motor = 1

            motor_points = motor_data[:, 0:6]
            motor_labels = motor_data[:, 6]  # result is a np array
            num_eachtype_in_one_motor, _ = np.histogram(motor_labels, bins=self.num_class, range=(
                0, self.num_class))  # count how much points is there for each type(usage of np.histotram)
            label_num_eachtype += num_eachtype_in_one_motor
            self.motors_points.append(motor_points)
            self.motors_labels.append(motor_labels)
            self.type_for_each_motor.append(class_motor)
            self.cover_exitence_for_each_motor.append(cover_exitence_motor)
            num_points_eachmotor.append(motor_labels.size)
        ######## make the goals  record goals ############
        # self.goals,self.mask=find_goals_kmeans(np.array(self.motors_points),np.array(self.motors_labels))
        # self.goals=self.goals.reshape(self.goals.shape[0],-1)
        # np.savetxt("goals.txt",self.goals.reshape(self.goals.shape[0],-1))
        # np.savetxt("mask.txt",self.mask)
        # if "Dataset3" in data_root:
        self.goals = np.loadtxt("kernels/goals3.txt")
        self.goals = self.goals.reshape(self.goals.shape[0], -1, 3)
        self.mask = np.loadtxt("kernels/mask3.txt")
        # elif "Dataset4" in data_root:
        #     self.goals=np.loadtxt("goals4.txt")
        #     self.goals=self.goals.reshape(self.goals.shape[0],-1,3)
        #     self.mask=np.loadtxt("mask4.txt") 
        # else:
        # self.goals=np.loadtxt("goals_finetune.txt")
        # self.goals=self.goals.reshape(self.goals.shape[0],-1,3)
        # self.mask=np.loadtxt("mask_finetune.txt")
        ############################################################################################

        ###########according to lable_num_eachmotor and bolt_weight, caculate the labelweights######
        label_num_eachtype = label_num_eachtype.astype(np.float32)
        label_num_eachtype__ = label_num_eachtype + 0
        self.persentage = label_num_eachtype__ / np.sum(label_num_eachtype__)
        print(self.persentage)
        ####reversed order
        label_num_eachtype[-1] /= bolt_weights
        labelweights = label_num_eachtype / np.sum(label_num_eachtype)

        labelweights = np.power(np.max(labelweights) / labelweights, 1 / 3)
        ##########################
        self.num_eachtype__ = label_num_eachtype__
        self.num_eachtype = label_num_eachtype
        self.labelweights = labelweights / np.sum(labelweights)
        ############################################################################################

        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########
        sample_prob_eachmotor = num_points_eachmotor / np.sum(
            num_points_eachmotor)  # probability for choosing from a specific motor
        num_interation = sample_rate * np.sum(
            num_points_eachmotor) / self.num_points  # num_of_all to choose npoints cloud
        self.motors_indes = []  # initial motors_indes list
        for index in range(len(num_points_eachmotor)):  # allocate the index according to probability
            sample_times_to_onemotor = int(round(sample_prob_eachmotor[index] * num_interation))
            motor_indes_onemotor = [index] * sample_times_to_onemotor
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################

    def __getitem__(self, index):

        points = self.motors_points[self.motors_indes[index]][:, 0:3]  # initialize the parameter
        labels = self.motors_labels[self.motors_indes[index]]
        type_for_each_motor = self.type_for_each_motor[self.motors_indes[index]]
        cover_exitence_for_each_motor = self.cover_exitence_for_each_motor[self.motors_indes[index]]
        n_points = points.shape[0]
        ########################have a randow choose of points from points cloud#######################
        choice = np.random.choice(n_points, self.num_points, replace=True)
        chosed_points = points[choice, :]
        chosed_labels = labels[choice]
        ###############################################################################################

        # return chosed_points,chosed_labels,class_for_each_motor,cover_exitence_for_each_motor
        return chosed_points, chosed_labels, type_for_each_motor, self.goals[self.motors_indes[index]], self.mask[
            self.motors_indes[index]], cover_exitence_for_each_motor

    def __len__(self):
        return len(self.motors_indes)


class MotorDataset_validation(Dataset):
    def __init__(self, split='train', data_root='directory to training data', num_class=6, num_points=4096,
                 test_area='Validation', block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_points = num_points
        self.block_size = block_size
        self.transform = transform
        motor_list = sorted(os.listdir(data_root))  # list all subdirectory
        # motor_filter=[motor for motor in motor_list if 'Type' in motor]     #filter all the file, whose name has no Type
        if split == 'train':  # load training files or validation files
            motor_positions = [motor for motor in motor_list if '{}'.format(test_area) not in motor]
        else:
            motor_positions = [motor for motor in motor_list if '{}'.format(test_area) in motor]
        ######################load the np file###################################################    
        self.motors_points = []  # initial object_motor_points and object_motor_lables
        self.motors_labels = []
        num_points_eachmotor = []  # initial a list to count the num of points for each motor
        self.type_for_each_motor = []
        self.cover_exitence_for_each_motor = []

        label_num_eachtype = np.zeros(num_class)  # initial a array to count how much points is there for each type
        for motor_position in tqdm(motor_positions, total=len(motor_positions)):
            motor_directory = os.path.join(data_root, motor_position)
            motor_data = np.load(motor_directory)
            if 'A0' in motor_position:
                class_motor = 0
            elif 'A1' in motor_position:
                class_motor = 1
            elif 'A2' in motor_position:
                class_motor = 2
            elif 'B0' in motor_position:
                class_motor = 3
            else:
                class_motor = 4

            if 'gear' in motor_position:
                cover_exitence_motor = 0
            else:
                cover_exitence_motor = 1

            motor_points = motor_data[:, 0:6]
            motor_labels = motor_data[:, 6]  # result is a np array
            num_eachtype_in_one_motor, _ = np.histogram(motor_labels, bins=num_class, range=(
                0, num_class))  # count how much points is there for each type(usage of np.histotram)
            label_num_eachtype += num_eachtype_in_one_motor
            self.motors_points.append(motor_points)
            self.motors_labels.append(motor_labels)
            self.type_for_each_motor.append(class_motor)
            self.cover_exitence_for_each_motor.append(cover_exitence_motor)
            num_points_eachmotor.append(motor_labels.size)
        ############################################################################################

        ###########according to lable_num_eachmotor and bolt_weight, caculate the labelweights######
        label_num_eachtype[-1] /= bolt_weights  # here to change the bolt weights
        labelweights = label_num_eachtype / np.sum(label_num_eachtype)
        labelweights = np.power(np.max(labelweights) / labelweights, 1 / 3)
        self.labelweight = labelweights / np.sum(labelweights)
        ############################################################################################

        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########
        sample_prob_eachmotor = num_points_eachmotor / np.sum(
            num_points_eachmotor)  # probability for choosing from a specific motor
        num_interation = sample_rate * np.sum(
            num_points_eachmotor) / self.num_points  # num_of_all to choose npoints cloud
        self.motors_indes = []  # initial motors_indes list
        for index in range(len(num_points_eachmotor)):  # allocate the index according to probability
            sample_times_to_onemotor = int(round(sample_prob_eachmotor[index] * num_interation))
            motor_indes_onemotor = [index] * sample_times_to_onemotor
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################

    def __getitem__(self, index):

        points = self.motors_points[self.motors_indes[index]][:, 0:3]  # initialize the parameter
        labels = self.motors_labels[self.motors_indes[index]]
        type_for_each_motor = self.type_for_each_motor[self.motors_indes[index]]
        cover_exitence_for_each_motor = self.cover_exitence_for_each_motor[self.motors_indes[index]]
        n_points = points.shape[0]
        # points=rotate(points)
        # normalized_points=pc_normalize(points)      #normalize the points

        ########################have a randow choose of points from points cloud#######################
        choice = np.random.choice(n_points, self.num_points, replace=True)
        chosed_points = points[choice, :]
        chosed_labels = labels[choice]

        ###############################################################################################

        return chosed_points, chosed_labels, type_for_each_motor, cover_exitence_for_each_motor

    def __len__(self):
        return len(self.motors_indes)


class MotorDataset_patch(Dataset):
    def __init__(self, split='train', data_root='directory to training data', num_points=4096, test_area='Validation',
                 block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.add = 0
        self.num_points = num_points
        self.block_size = block_size
        self.transform = transform
        motor_list = sorted(os.listdir(data_root))  # list all subdirectory
        # motor_filter=[motor for motor in motor_list if 'Training' in motor]     #filter all the file, whose name has no Type      
        if split == 'train':  # load training files or validation files
            motor_positions = [motor for motor in motor_list if '{}'.format(test_area) not in motor]
        else:
            motor_positions = [motor for motor in motor_list if '{}'.format(test_area) in motor]

        ######################load the np file###################################################

        self.motors_points = []  # initial object_motor_points and object_motor_lables
        self.motors_labels = []
        self.interation_times_eachmotor = []
        self.type_for_each_motor = []
        self.cover_exitence_for_each_motor = []

        label_num_eachtype = np.zeros(6)  # initial a array to count how much points is there for each type
        for motor_position in tqdm(motor_positions, total=len(motor_positions)):
            motor_directory = os.path.join(data_root, motor_position)
            motor_data = np.load(motor_directory)
            if 'A0' in motor_position:
                class_motor = 0
            elif 'A1' in motor_position:
                class_motor = 1
            elif 'A2' in motor_position:
                class_motor = 2
            elif 'B0' in motor_position:
                class_motor = 3
            else:
                class_motor = 4
            if 'gear' in motor_position:
                cover_exitence_motor = 0
            else:
                cover_exitence_motor = 1
            motor_points = motor_data[:, 0:3]
            motor_points = pc_normalize(motor_points)  # normalize the points
            motor_labels = motor_data[:, 6]
            self.type_for_each_motor.append(class_motor)
            self.cover_exitence_for_each_motor.append(cover_exitence_motor)
            motor_points_labels = []

            ################       judge if current motor points cloud can just include integel             ########
            ###############     times of sub points_clouds of num_point,if not, patch it to integel one     ########
            current_motor_size = motor_points.shape[0]
            if current_motor_size % self.num_points != 0:
                num_add_points = self.num_points - (current_motor_size % self.num_points)
                choice = np.random.choice(current_motor_size, num_add_points,
                                          replace=True)  # pick out some points from current cloud to patch up the current cloud
                add_points = motor_points[choice, :]
                motor_points = np.vstack((motor_points, add_points))
                add_labels = motor_labels[choice]
                motor_labels = np.hstack((motor_labels, add_labels))
            #########################################################################################################
            #########################################################################################################

            motor_points_labels = np.hstack((motor_points, motor_labels.reshape(
                (motor_labels.size, 1))))  # merge the labels and points in order to schuffle it
            np.random.shuffle(motor_points_labels)
            motor_points = motor_points_labels[:, 0:3]  # get the schuffled points and lables
            motor_labels = motor_points_labels[:, 3]
            self.interation_times_eachmotor.append(
                motor_labels.size / self.num_points)  # record how money 4096 points could be taken out for one motor points cloud after patch
            num_eachtype_in_one_motor, _ = np.histogram(motor_labels, bins=6, range=(
                0, 6))  # count how much points is there for each type(usage of np.histotram)
            label_num_eachtype += num_eachtype_in_one_motor
            self.motors_points.append(motor_points)
            self.motors_labels.append(motor_labels)
        ############################################################################################

        ###########according to lable_num_eachmotor and bolt_weight, caculate the labelweights######
        label_num_eachtype[-1] /= bolt_weights
        labelweights = label_num_eachtype / np.sum(label_num_eachtype)
        labelweights = np.power(np.max(labelweights) / labelweights, 1 / 3)
        self.labelweight = labelweights / np.sum(labelweights)
        ############################################################################################

        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########
        self.motors_indes = []  # initial motors_indes list
        for index in range(len(self.interation_times_eachmotor)):  # allocate the index according to probability
            motor_indes_onemotor = [index] * int(self.interation_times_eachmotor[index])
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################

        #####################################   set the dictionary for dataloader index according to motors_points structure        ########
        self.dic_block_accumulated_per_motors = {}
        key = 0
        for index in range(len(self.interation_times_eachmotor)):
            if index != 0:
                key = key + self.interation_times_eachmotor[index - 1]
            for num_clouds_per_motor in range(int(self.interation_times_eachmotor[index])):
                self.dic_block_accumulated_per_motors[int(key + num_clouds_per_motor)] = num_clouds_per_motor
        ####################################################################################################################################

    def __getitem__(self, index):
        points = self.motors_points[self.motors_indes[index]]  # initialize the points cloud for each motor
        labels = self.motors_labels[self.motors_indes[index]]
        type_for_each_motor = self.type_for_each_motor[self.motors_indes[index]]
        cover_existence = self.cover_exitence_for_each_motor[self.motors_indes[index]]
        sequence = np.arange(self.num_points)
        chosed_points = points[self.num_points * self.dic_block_accumulated_per_motors[index] + sequence,
                        :]  # ensure all the points could be picked out by the ways of patch
        chosed_labels = labels[self.num_points * self.dic_block_accumulated_per_motors[index] + sequence]
        self.add = self.add + index
        return chosed_points, chosed_labels, type_for_each_motor, cover_existence

    def __len__(self):
        return len(self.motors_indes)
