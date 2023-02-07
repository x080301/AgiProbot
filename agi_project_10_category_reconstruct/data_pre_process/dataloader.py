"""
@Author: bixuelei
@Contact: bxueleibi@gmial.com
@File: dataloader.py
@Time: 2022/1/16 3:49 PM
"""

import os
import numpy as np
import random
from numpy.random import choice
from tqdm import tqdm           #used to display the circulation position, to see where the code is running at
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
# from display import Visuell_PointCloud,Visuell,Visuell__
import torch
# from kmeans_pytorch import kmeans


####### normalize the point cloud############
def pc_normalize(pc):
    centroid=np.mean(pc,axis=0)
    pc=pc-centroid
    max_distance=np.sqrt(np.max(np.sum(pc**2,axis=1)))
    pc=pc/max_distance
    return pc

class MotorDataset_patch(Dataset):
    def __init__(self,points):
        super().__init__()
        self.num_points=2048     
     
        self.motors_points=[]       #initial object_motor_points and object_motor_lables
        self.interation_times_eachmotor = []
        motor_points=points[:,0:3]
        motor_points=pc_normalize(motor_points)      #normalize the points
        motor_points=np.hstack((motor_points,points[:,0:3])) 

        current_motor_size=motor_points.shape[0]
        if current_motor_size % self.num_points !=0:
            num_add_points=self.num_points-(current_motor_size % self.num_points)
            choice=np.random.choice(current_motor_size,num_add_points,replace=True)     #pick out some points from current cloud to patch up the current cloud
            add_points=motor_points[choice,:]
            motor_points=np.vstack((motor_points,add_points))
        np.random.shuffle(motor_points)
        self.interation_times_eachmotor.append(current_motor_size/self.num_points)       #record how money 4096 points could be taken out for one motor points cloud after patch
        self.motors_points.append(motor_points)




        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########      
        self.motors_indes=[]        #initial motors_indes list    
        for index in range(len(self.interation_times_eachmotor)):      #allocate the index according to probability
            motor_indes_onemotor=[index]*int(self.interation_times_eachmotor[index])
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################


        #####################################   set the dictionary for dataloader index according to motors_points structure        ########
        self.dic_block_accumulated_per_motors={}
        key=0
        for index in range(len(self.interation_times_eachmotor)):
            if index!=0:
                key=key+self.interation_times_eachmotor[index-1]
            for num_clouds_per_motor in range(int(self.interation_times_eachmotor[index])):
                self.dic_block_accumulated_per_motors[int(key+num_clouds_per_motor)]=num_clouds_per_motor
        ####################################################################################################################################


    def __getitem__(self,index):   
        points=self.motors_points[self.motors_indes[index]]      #initialize the points cloud for each motor
        sequence=np.arange(self.num_points)
        chosed_points=points[self.num_points*self.dic_block_accumulated_per_motors[index]+sequence,:]       #ensure all the points could be picked out by the ways of patch
        return chosed_points[:,0:3],chosed_points[:,3:6]

    def __len__(self):                                                                            
        return len(self.motors_indes)