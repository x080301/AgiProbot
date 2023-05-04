"""
@Author: bixuelei
@Contact: bxueleibi@gmial.com
@File: data_loader.py
@Time: 2022/1/16 3:49 PM
"""

import os
import numpy as np

from tqdm import tqdm  # used to display the circulation position, to see where the code is running at
from torch.utils.data import Dataset


class MotorDataset(Dataset):
    def __init__(self, mode='train', data_dir='directory to training data', num_class=2, num_points=4096,
                 test_area='Validation', sample_rate=1.0):

        super().__init__()
        self.num_points = num_points
        self.num_class = num_class
        motor_list = sorted(os.listdir(data_dir))  # list all subdirectory

        if mode == 'train':  # load training files or validation files
            motor_positions = [motor for motor in motor_list if '{}'.format(test_area) not in motor]
        else:
            motor_positions = [motor for motor in motor_list if '{}'.format(test_area) in motor]

        ######################load the np file###################################################

        points_motors = []  # initial object_motor_points and object_motor_lables
        labels_motors = []
        points_num_eachmotor = []  # initial a list to count the num of points for each motor
        label_num_eachtype = np.zeros(self.num_class)  # initial a array to count how much points is there for each type

        for motor_position in tqdm(motor_positions, total=len(motor_positions)):
            motor_directory = os.path.join(data_dir, motor_position)
            motor_data = np.load(motor_directory)

            points_in_one_motor = motor_data[:, 0:6]
            labels_in_one_motor = motor_data[:, 6]  # result is a np array
            points_motors.append(points_in_one_motor)
            labels_motors.append(labels_in_one_motor)

            points_num_eachmotor.append(points_in_one_motor.shape[0])

            num_eachtype_in_one_motor, _ = np.histogram(labels_in_one_motor, bins=self.num_class, range=(
                0, self.num_class))  # count how much points is there for each type(usage of np.histotram)
            label_num_eachtype += num_eachtype_in_one_motor

        ###########according to lable_num_eachmotor, caculate the labelweights######
        label_num_eachtype = label_num_eachtype.astype(np.float32)
        label_num_eachtype__ = label_num_eachtype + 0
        persentage = label_num_eachtype__ / np.sum(label_num_eachtype__)
        print(persentage)
        ####reversed order
        label_weights = label_num_eachtype / np.sum(label_num_eachtype)
        label_weights = np.power(np.max(label_weights) / label_weights, 1 / 3)
        label_weights = label_weights / np.sum(label_weights)
        ############################################################################################

        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########
        sample_prob_eachmotor = points_num_eachmotor / np.sum(
            points_num_eachmotor)  # probability for choosing from a specific motor
        num_interation = sample_rate * np.sum(
            points_num_eachmotor) / self.num_points  # num_of_all to choose npoints cloud
        motors_indes = []  # initial motors_indes list
        for index in range(len(points_num_eachmotor)):  # allocate the index according to probability
            sample_times_to_onemotor = int(round(sample_prob_eachmotor[index] * num_interation))
            motor_indes_onemotor = [index] * sample_times_to_onemotor
            motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################

        self.points_motors = points_motors
        self.labels_motors = labels_motors
        self.persentage_each_type = persentage
        self.label_weights = label_weights
        self.motors_indes = motors_indes

    def __getitem__(self, index):

        points = self.points_motors[self.motors_indes[index]][:, 0:3]  # initialize the parameter
        labels = self.labels_motors[self.motors_indes[index]]
        ########################have a randow choose of points from points cloud#######################
        choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]
        segmentation_labels = labels[choice]
        ###############################################################################################
        points = np.asarray(points)
        segmentation_labels = np.asarray(segmentation_labels)

        return points, segmentation_labels
        '''
        return points, segmentation_labels, motor_type_classification_labels, self.goals[self.motors_indes[index]], \
               self.mask[self.motors_indes[index]], cover_exitence_for_each_motor
        # (points, target, type_label, goals, masks, type)
        '''

    def __len__(self):
        return len(self.motors_indes)
