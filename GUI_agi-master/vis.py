# import jinja2
from fileinput import filename
from turtle import pos
import numpy as np
import argparse
from sklearn.metrics import coverage_error
import torch
from torch.utils.data import DataLoader
from dataloader import MotorDataset_patch
import torch.nn as nn
import random
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
import sys
import vtk
import csv
from PyQt5 import QtCore, QtGui, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from gui import Ui_MainWindow
import math
from sklearn.cluster import DBSCAN
import open3d as o3d
import os
from pipeline.model_rotation import PCT_semseg
from display import vis
from tqdm import tqdm
########### used to transfer the rgb to r g b###########3
from struct import pack, unpack

cam_to_base_transform = [[6.3758686e-02, 9.2318553e-01, -3.7902945e-01, 4.5398907e+01],
                         [9.8811066e-01, -5.1557920e-03, 1.5365793e-01, -7.5876160e+02],
                         [1.3990058e-01, -3.8432005e-01, -9.1253817e-01, 9.6543054e+02],
                         [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]

# color_map={"back_ground":[0,0,128],
#            "cover":[0,100,0],
#            "gear_container":[0,255,0],
#            "charger":[255,255,0],
#            "bottom":[255,165,0],
#            "bolts":[255,0,0]}

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


def Read_PCD(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    return np.concatenate([points, colors], axis=-1)


class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Mywindow, self).__init__()
        # pcd=Read_PCD("/home/bi/study/thesis/pyqt/test.pcd")
        self.fileName = ""
        self.type = -1
        self.state = 0
        self.points_to_model = []
        self.setupUi(self)
        self.setWindowTitle('Visialization of result')

        self.frame = QtWidgets.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)  # give a QVTK a Qt framework
        self.formLayout_1.addWidget(self.vtkWidget)  # connect the vtkWidget with a formlayout.
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        self.ren.SetBackground(192 / 255, 192 / 255, 192 / 255)
        self.ren.ResetCamera()

        self.frame__ = QtWidgets.QFrame()
        self.vtkWidget__ = QVTKRenderWindowInteractor(self.frame__)  # give a QVTK a Qt framework
        self.formLayout_2.addWidget(self.vtkWidget__)  # connect the vtkWidget with a formlayout.
        self.ren__ = vtk.vtkRenderer()
        self.vtkWidget__.GetRenderWindow().AddRenderer(self.ren__)
        self.iren__ = self.vtkWidget__.GetRenderWindow().GetInteractor()
        self.iren__.Initialize()
        self.ren__.SetBackground(192 / 255, 192 / 255, 192 / 255)
        self.ren__.ResetCamera()

        self.action_load.triggered.connect(self.load)
        self.action_display_cuboid_pc.triggered.connect(self.cuboid_display__)
        self.action_display_original_pc.triggered.connect(self.display_original)
        self.action_cubiod_prediction.triggered.connect(self.predict_cropped)
        self.action_bolts_position.triggered.connect(self.filter_bolts)
        self.action_motor_orientation.triggered.connect(self.estimate_cover)
        self.actionGear_position.triggered.connect(self.display_gear)
        self.actionsave.triggered.connect(self.save)
        self.pushButton.clicked.connect(self.predict)
        self.show()

    def load(self):
        self.cuted = 0
        self.ren.Clear()
        self.ren.NewInstance()
        # Create source
        self.fileName_previous = self.fileName
        self.label.setText("Loading the file")
        self.label.adjustSize()
        self.label.repaint()
        self.fileName = \
        QFileDialog.getOpenFileName(self, caption="choose the file you want to predict", filter="*.pcd *.ply")[0]
        self.filename_ = self.fileName.split('/')[-1]
        # cloud_=Read_PCD(fileName)
        if self.fileName_previous != self.fileName:
            if self.fileName_previous != "":
                self.ren.RemoveActor(self.actor)
            self.cloud = Read_PCD(self.fileName)
            # self.cloud = pcl.load_XYZRGB(self.fileName)
            self.judge_cut_again = "yes"
            self.judge_predict_again = "yes"
            self.trans_again = "yes"
            self.search_bolts_again = "yes"
            self.estimate_again = "yes"
        self.num_points = len(self.cloud)

        # create the geometry of a point
        self.points = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        # create the topology of the point
        self.vertices = vtk.vtkCellArray()

        # Setup colors
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")

        ############display the points
        if self.fileName_previous != self.fileName:
            self.points_to_model = []
            for i in range(self.num_points):
                dp = self.cloud[i]
                # if dp[2]<0 or dp[2]>800:
                #   continue
                # self.points_to_model.append([dp[0], dp[1], dp[2],dp[3]])
                id = self.points.InsertNextPoint(dp[0], dp[1], dp[2])
                self.vertices.InsertNextCell(1)
                self.vertices.InsertCellPoint(id)

                r = int(self.cloud[i][3] * 255)
                g = int(self.cloud[i][4] * 255)
                b = int(self.cloud[i][5] * 255)
                self.points_to_model.append([dp[0], dp[1], dp[2], r, g, b])
                self.Colors.InsertNextTuple3(r, g, b)
        else:
            for i in range(self.num_points):
                id = self.points.InsertNextPoint(self.points_to_model[i][0], self.points_to_model[i][1],
                                                 self.points_to_model[i][2])
                self.vertices.InsertNextCell(1)
                self.vertices.InsertCellPoint(id)
                self.Colors.InsertNextTuple3(self.points_to_model[i][3], self.points_to_model[i][4],
                                             self.points_to_model[i][5])

        num_points = str(self.num_points)
        self.label_3.setText(num_points)
        self.label_3.adjustSize()
        self.label.setText("Original pc")
        self.label.adjustSize()

        ##VTK color representation
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.SetVerts(self.vertices)
        self.polydata.GetPointData().SetScalars(self.Colors)
        self.polydata.Modified()

        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.SetInputData(self.polydata)
        self.glyphFilter.Update()

        self.dataMapper = vtk.vtkPolyDataMapper()
        self.dataMapper.SetInputConnection(self.glyphFilter.GetOutputPort())

        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.dataMapper)

        self.ren.AddActor(self.actor)
        self.ren.ResetCamera()

    def display_original(self):
        self.label.setText("Loading original pc")
        self.label.adjustSize()
        self.label.repaint()
        self.ren.RemoveActor(self.actor)

        # create the geometry of a point
        self.points = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        # create the topology of the point
        self.vertices = vtk.vtkCellArray()

        # Setup colors
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")

        ############display the points
        for i in range(int(self.num_points)):
            id = self.points.InsertNextPoint(self.points_to_model[i][0], self.points_to_model[i][1],
                                             self.points_to_model[i][2])
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(id)
            self.Colors.InsertNextTuple3(self.points_to_model[i][3], self.points_to_model[i][4],
                                         self.points_to_model[i][5])

        num_points = str(self.num_points)
        self.label_3.setText(num_points)
        self.label_3.adjustSize()
        self.label.setText("Original pc")
        self.label.adjustSize()
        self.cuted = 1

        ##VTK color representation
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.SetVerts(self.vertices)
        self.polydata.GetPointData().SetScalars(self.Colors)
        self.polydata.Modified()

        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.SetInputData(self.polydata)
        self.glyphFilter.Update()

        self.dataMapper = vtk.vtkPolyDataMapper()
        self.dataMapper.SetInputConnection(self.glyphFilter.GetOutputPort())

        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.dataMapper)

        self.ren.AddActor(self.actor)
        self.ren.ResetCamera()

    def cut_cuboid(self):
        if self.judge_cut_again == self.fileName:
            return
        # self.label.setText("cutting the cuboid")
        # self.label.adjustSize()
        # ########### patch the cloud popints
        self.label.setText("Start to cut cuboid")
        self.label.adjustSize()
        self.label.repaint()
        self.points_to_model = np.array(self.points_to_model)
        self.motor_scene, self.residual_scene = cut_motor(self.points_to_model)
        current_points_size = self.motor_scene.shape[0]
        if current_points_size % 2048 != 0:
            num_add_points = 2048 - (current_points_size % 2048)
            choice = np.random.choice(current_points_size, num_add_points, replace=True)
            add_points = self.motor_scene[choice, :]
            self.motor_points = np.vstack((self.motor_scene, add_points))
            np.random.shuffle(self.motor_points)
        else:
            self.motor_points = self.motor_scene
            np.random.shuffle(self.motor_points)
        self.judge_cut_again = self.fileName
        self.num_point_cut = self.motor_points.shape[0]

    def cuboid_display(self):

        # if self.judge_cut_again!=self.fileName:
        self.cut_cuboid()
        self.judge_cut_again = self.fileName
        self.ren.RemoveActor(self.actor)
        # create the geometry of a point
        self.points = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        # create the topology of the point
        self.vertices = vtk.vtkCellArray()

        # Setup colors
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")

        self.motor_points_trans = np.array(camera_to_base(self.motor_points[:, 0:3]))
        for i in range(self.motor_points_trans.shape[0]):
            id = self.points.InsertNextPoint(self.motor_points_trans[i][0], self.motor_points_trans[i][1],
                                             self.motor_points_trans[i][2])
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(id)
            self.Colors.InsertNextTuple3(self.motor_points[i][3], self.motor_points[i][4], self.motor_points[i][5])

        self.label.setText("Cropped cuboid")
        self.label.adjustSize()
        self.num_point_cut = str(self.num_point_cut)
        self.label_3.setText(self.num_point_cut)
        self.label_3.adjustSize()

        ##VTK color representation
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.SetVerts(self.vertices)
        self.polydata.GetPointData().SetScalars(self.Colors)
        self.polydata.Modified()

        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.SetInputData(self.polydata)
        self.glyphFilter.Update()

        self.dataMapper = vtk.vtkPolyDataMapper()
        self.dataMapper.SetInputConnection(self.glyphFilter.GetOutputPort())

        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.dataMapper)

        self.ren.AddActor(self.actor)
        self.ren.ResetCamera()

    def cuboid_display__(self):

        # if self.judge_cut_again!=self.fileName:
        self.cut_cuboid()
        self.judge_cut_again = self.fileName
        self.ren.RemoveActor(self.actor)
        # create the geometry of a point
        self.points = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        # create the topology of the point
        self.vertices = vtk.vtkCellArray()

        # Setup colors
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")

        # for i in range(self.motor_points.shape[0]):
        #   dp = self.motor_points[i]
        #   id=self.points.InsertNextPoint(dp[0], dp[1], dp[2])
        #   self.vertices.InsertNextCell(1)
        #   self.vertices.InsertCellPoint(id)
        #   inter=unpack('i',pack('f',self.motor_points[i][3]))[0]
        #   r=(inter>>16) & 0x0000ff
        #   g=(inter>>8) & 0x0000ff
        #   b=(inter>>0) & 0x0000ff
        #   self.Colors.InsertNextTuple3(r, g, b)
        self.motor_points_trans = np.array(camera_to_base(self.motor_points[:, 0:3]))
        for i in range(self.motor_points_trans.shape[0]):
            id = self.points.InsertNextPoint(self.motor_points_trans[i][0], self.motor_points_trans[i][1],
                                             self.motor_points_trans[i][2])
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(id)
            self.Colors.InsertNextTuple3(self.motor_points[i][3], self.motor_points[i][4], self.motor_points[i][5])

        self.label.setText("Cropped cuboid")
        self.label.adjustSize()
        self.num_point_cut = str(self.num_point_cut)

        ##VTK color representation
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.SetVerts(self.vertices)
        self.polydata.GetPointData().SetScalars(self.Colors)
        self.polydata.Modified()

        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.SetInputData(self.polydata)
        self.glyphFilter.Update()

        self.dataMapper = vtk.vtkPolyDataMapper()
        self.dataMapper.SetInputConnection(self.glyphFilter.GetOutputPort())

        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.dataMapper)

        self.ren.AddActor(self.actor)
        self.ren.ResetCamera()

    def predict(self):
        if self.judge_cut_again != self.fileName:
            self.cut_cuboid()
            self.judge_cut_again = self.fileName
        else:
            self.predict_cropped
            return

        self.cuboid_display__()

        if self.judge_predict_again != self.fileName:
            self.label_2.setText("Send to neural network to predict")
            self.label_2.adjustSize()
            self.label_2.repaint()
            # self.label_3.setText(str(self.num_point_cut))
            # self.label_3.adjustSize()
            self.motor_points_forecast, self.type = predict(self.motor_points[:, 0:3])
        # self.label_3.setText(str(self.num_point_cut))
        # self.label_3.adjustSize()
        self.judge_predict_again = self.fileName
        if self.cuted != 2:
            self.cuted = 1
        self.transfer_to_borot_coordinate()
        self.label.setText("Cropped cuboid")
        self.label.adjustSize()

    def predict_cropped(self):
        # if self.judge_cut_again!=self.fileName:
        #  self.cut_cuboid()
        #   self.judge_cut_again=self.fileName

        self.cuboid_display__()

        if self.judge_predict_again != self.fileName:
            self.label_2.setText("Send to neural network to predict")
            self.label_2.adjustSize()
            self.label_2.repaint()
            # self.label_3.setText(str(self.num_point_cut))
            # self.label_3.adjustSize()
            self.motor_points_forecast, self.type = predict(self.motor_points[:, 0:3])
        # self.label_3.setText(str(self.num_point_cut))
        # self.label_3.adjustSize()
        self.judge_predict_again = self.fileName
        self.transfer_to_borot_coordinate__()
        self.label.setText("Cropped cuboid")
        self.label.adjustSize()

    def clear(self):
        self.label_2.setText("")
        self.label_2.adjustSize()
        self.label_6.setText("")
        self.label_6.adjustSize()
        self.label_11.setText("")
        self.label_11.adjustSize()
        self.label_5.setText("")
        self.label_5.adjustSize()
        self.label_12.setText("")
        self.label_12.adjustSize()
        self.label_7.setText("")
        self.label_7.adjustSize()
        self.label_13.setText("")
        self.label_13.adjustSize()
        self.label_8.setText("")
        self.label_8.adjustSize()
        self.label_14.setText("")
        self.label_14.adjustSize()
        self.label_9.setText("")
        self.label_9.adjustSize()
        self.label_15.setText("")
        self.label_15.adjustSize()
        self.label_10.setText("")
        self.label_10.adjustSize()
        self.label_16.setText("")
        self.label_16.adjustSize()

    def transfer_to_borot_coordinate(self):
        self.label_2.setText("Start to visiualise result")
        self.label_2.adjustSize()
        self.label_2.repaint()
        if self.fileName_previous != "":
            self.ren__.RemoveActor(self.actor__)
        self.fileName_previous = "been precessed"
        # create the geometry of a point
        self.points__ = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        # create the topology of the point
        self.vertices__ = vtk.vtkCellArray()

        # Setup colors
        self.Colors__ = vtk.vtkUnsignedCharArray()
        self.Colors__.SetNumberOfComponents(3)
        self.Colors__.SetName("Colors__")
        if self.trans_again != self.fileName:
            self.motor_points_forecast_in_robot = np.random.rand(self.motor_points_forecast.shape[0], 4)
            self.motor_points_forecast_in_robot[:, 0:3] = np.array(camera_to_base(self.motor_points_forecast[:, 0:3]))
            self.motor_points_forecast_in_robot[:, 3] = np.array(self.motor_points_forecast[:, 3])
        self.trans_again = self.fileName
        self.cover_existence, self.covers, self.normal = find_covers(self.motor_points_forecast_in_robot)
        if self.cover_existence > 0:
            self.filter_bolts_but_not_to_show()
        else:
            self.find_gear_but_not_to_show()
        for i in range(self.motor_points_forecast_in_robot.shape[0]):
            dp = self.motor_points_forecast_in_robot[i]
            id = self.points__.InsertNextPoint(dp[0], dp[1], dp[2])
            self.vertices__.InsertNextCell(1)
            self.vertices__.InsertCellPoint(id)
            if dp[3] == 0:
                r = color_map["back_ground"][0]
                g = color_map["back_ground"][1]
                b = color_map["back_ground"][2]
            elif dp[3] == 1:
                r = color_map["cover"][0]
                g = color_map["cover"][1]
                b = color_map["cover"][2]
            elif dp[3] == 2:
                r = color_map["gear_container"][0]
                g = color_map["gear_container"][1]
                b = color_map["gear_container"][2]
            elif dp[3] == 3:
                r = color_map["charger"][0]
                g = color_map["charger"][1]
                b = color_map["charger"][2]
            elif dp[3] == 4:
                r = color_map["bottom"][0]
                g = color_map["bottom"][1]
                b = color_map["bottom"][2]
            elif dp[3] == 5:
                r = color_map["side_bolts"][0]
                g = color_map["side_bolts"][1]
                b = color_map["side_bolts"][2]
            elif dp[3] == 6:
                r = color_map["bolts"][0]
                g = color_map["bolts"][1]
                b = color_map["bolts"][2]
            elif dp[3] == 8:
                r = color_map["upgear_a"][0]
                g = color_map["upgear_a"][1]
                b = color_map["upgear_a"][2]
            elif dp[3] == 7:
                r = color_map["lowgear_a"][0]
                g = color_map["lowgear_a"][1]
                b = color_map["lowgear_a"][2]
            else:
                r = color_map["gear_b"][0]
                g = color_map["gear_b"][1]
                b = color_map["gear_b"][2]
            self.Colors__.InsertNextTuple3(r, g, b)
        ## VTK color representation   22222222
        polydata__ = vtk.vtkPolyData()
        polydata__.SetPoints(self.points__)
        polydata__.SetVerts(self.vertices__)
        polydata__.GetPointData().SetScalars(self.Colors__)
        polydata__.Modified()

        glyphFilter__ = vtk.vtkVertexGlyphFilter()
        glyphFilter__.SetInputData(polydata__)
        glyphFilter__.Update()

        dataMapper__ = vtk.vtkPolyDataMapper()
        dataMapper__.SetInputConnection(glyphFilter__.GetOutputPort())

        # Create an actor
        self.actor__ = vtk.vtkActor()
        self.actor__.SetMapper(dataMapper__)

        self.ren__.AddActor(self.actor__)
        self.ren__.ResetCamera()

        self.label_2.setText("Predicted result of cuboid")
        self.label_2.adjustSize()
        if self.cuted:
            self.label_3.setText(str(self.num_point_cut))
            self.label_3.adjustSize()
            if self.type <= 1:
                self.label_18.setText("Type A1")
                self.label_18.adjustSize()
            elif self.type == 2:
                self.label_18.setText("Type A2")
                self.label_18.adjustSize()
            else:
                self.label_18.setText("Type B")
                self.label_18.adjustSize()
            self.cuted = 2

    def transfer_to_borot_coordinate__(self):
        self.label_2.setText("Start to visiualise result")
        self.label_2.adjustSize()
        self.label_2.repaint()
        if self.fileName_previous != "":
            self.ren__.RemoveActor(self.actor__)
        self.fileName_previous = "been precessed"
        # create the geometry of a point
        self.points__ = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        # create the topology of the point
        self.vertices__ = vtk.vtkCellArray()

        # Setup colors
        self.Colors__ = vtk.vtkUnsignedCharArray()
        self.Colors__.SetNumberOfComponents(3)
        self.Colors__.SetName("Colors__")
        if self.trans_again != self.fileName:
            self.motor_points_forecast_in_robot = np.random.rand(self.motor_points_forecast.shape[0], 4)
            self.motor_points_forecast_in_robot[:, 0:3] = np.array(camera_to_base(self.motor_points_forecast[:, 0:3]))
            self.motor_points_forecast_in_robot[:, 3] = np.array(self.motor_points_forecast[:, 3])
        self.trans_again = self.fileName
        self.cover_existence, self.covers, self.normal = find_covers(self.motor_points_forecast_in_robot)
        if self.cover_existence > 0:
            self.filter_bolts_but_not_to_show()
        else:
            self.find_gear_but_not_to_show()
        for i in range(self.motor_points_forecast_in_robot.shape[0]):
            dp = self.motor_points_forecast_in_robot[i]
            id = self.points__.InsertNextPoint(dp[0], dp[1], dp[2])
            self.vertices__.InsertNextCell(1)
            self.vertices__.InsertCellPoint(id)
            if dp[3] == 0:
                r = color_map["back_ground"][0]
                g = color_map["back_ground"][1]
                b = color_map["back_ground"][2]
            elif dp[3] == 1:
                r = color_map["cover"][0]
                g = color_map["cover"][1]
                b = color_map["cover"][2]
            elif dp[3] == 2:
                r = color_map["gear_container"][0]
                g = color_map["gear_container"][1]
                b = color_map["gear_container"][2]
            elif dp[3] == 3:
                r = color_map["charger"][0]
                g = color_map["charger"][1]
                b = color_map["charger"][2]
            elif dp[3] == 4:
                r = color_map["bottom"][0]
                g = color_map["bottom"][1]
                b = color_map["bottom"][2]
            elif dp[3] == 5:
                r = color_map["side_bolts"][0]
                g = color_map["side_bolts"][1]
                b = color_map["side_bolts"][2]
            elif dp[3] == 6:
                r = color_map["bolts"][0]
                g = color_map["bolts"][1]
                b = color_map["bolts"][2]
            elif dp[3] == 8:
                r = color_map["upgear_a"][0]
                g = color_map["upgear_a"][1]
                b = color_map["upgear_a"][2]
            elif dp[3] == 7:
                r = color_map["lowgear_a"][0]
                g = color_map["lowgear_a"][1]
                b = color_map["lowgear_a"][2]
            else:
                r = color_map["gear_b"][0]
                g = color_map["gear_b"][1]
                b = color_map["gear_b"][2]
            self.Colors__.InsertNextTuple3(r, g, b)
        ## VTK color representation   22222222
        polydata__ = vtk.vtkPolyData()
        polydata__.SetPoints(self.points__)
        polydata__.SetVerts(self.vertices__)
        polydata__.GetPointData().SetScalars(self.Colors__)
        polydata__.Modified()

        glyphFilter__ = vtk.vtkVertexGlyphFilter()
        glyphFilter__.SetInputData(polydata__)
        glyphFilter__.Update()

        dataMapper__ = vtk.vtkPolyDataMapper()
        dataMapper__.SetInputConnection(glyphFilter__.GetOutputPort())

        # Create an actor
        self.actor__ = vtk.vtkActor()
        self.actor__.SetMapper(dataMapper__)

        self.ren__.AddActor(self.actor__)
        self.ren__.ResetCamera()

        self.label_2.setText("Predicted result of cuboid")
        self.label_2.adjustSize()
        if self.cuted == 1:
            self.label_3.setText(str(self.num_point_cut))
            self.label_3.adjustSize()
            self.cuted = 2

    def estimate_cover(self):
        if self.cover_existence < 0:
            self.clear()
            self.label_6.setText("Warning:")
            self.label_6.adjustSize()
            self.label_11.setText("Cover has been removed\nno cover points found")
            self.label_11.adjustSize()
            return
        self.ren__.RemoveActor(self.actor__)
        self.clear()
        # #self.covers,self.Rx_Ry_Rz=find_covers(self.motor_points_forecast)
        # if self.estimate_again!=self.fileName:
        #   _,self.covers,self.normal=find_covers(self.motor_points_forecast_in_robot)
        # self.estimate_again==self.fileName
        self.label_2.setText("Estimated orientation of bolts")
        self.label_2.adjustSize()
        xx = '[' + str(self.normal[0]) + ', ' + str(self.normal[1]) + ', ' + str(self.normal[2]) + ']'
        self.label_6.setText("Rx_Ry_Rz:")
        self.label_6.adjustSize()
        self.label_11.setText(xx)
        self.label_11.adjustSize()
        # create the geometry of a point
        self.points__ = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        # create the topology of the point
        self.vertices__ = vtk.vtkCellArray()

        # Setup colors
        self.Colors__ = vtk.vtkUnsignedCharArray()
        self.Colors__.SetNumberOfComponents(3)
        self.Colors__.SetName("Colors__")
        for i in range(self.covers.shape[0]):
            dp = self.covers[i]
            id = self.points__.InsertNextPoint(dp[0], dp[1], dp[2])
            self.vertices__.InsertNextCell(1)
            self.vertices__.InsertCellPoint(id)
            self.Colors__.InsertNextTuple3(0, 100, 0)

        ## VTK color representation   22222222
        polydata__ = vtk.vtkPolyData()
        polydata__.SetPoints(self.points__)
        polydata__.SetVerts(self.vertices__)
        polydata__.GetPointData().SetScalars(self.Colors__)
        polydata__.Modified()

        glyphFilter__ = vtk.vtkVertexGlyphFilter()
        glyphFilter__.SetInputData(polydata__)
        glyphFilter__.Update()

        dataMapper__ = vtk.vtkPolyDataMapper()
        dataMapper__.SetInputConnection(glyphFilter__.GetOutputPort())

        # Create an actor
        self.actor__ = vtk.vtkActor()
        self.actor__.SetMapper(dataMapper__)

        self.ren__.AddActor(self.actor__)
        self.ren__.ResetCamera()

    def filter_bolts(self):
        if self.cover_existence <= 0:
            self.clear()
            self.label_6.setText("Warning:")
            self.label_6.adjustSize()
            self.label_11.setText("Cover has been removed\nno cover screw found")
            self.label_11.adjustSize()
            return
        self.ren__.RemoveActor(self.actor__)
        if self.search_bolts_again != self.fileName:
            self.positions_bolts, self.num_bolts, self.bolts = find_bolts(self.motor_points_forecast_in_robot, eps=2.5,
                                                                          min_points=50)
        self.search_bolts_again = self.fileName

        self.label_2.setText("Estimated positions of bolts")
        self.label_2.adjustSize()
        self.label_6.setText("Number of cover screws:")
        self.label_6.adjustSize()
        self.label_11.setText(str(self.num_bolts))
        self.label_11.adjustSize()
        self.label_5.setText("")
        self.label_5.adjustSize()
        self.label_12.setText("")
        self.label_12.adjustSize()
        self.label_7.setText("")
        self.label_7.adjustSize()
        self.label_13.setText("")
        self.label_13.adjustSize()
        self.label_8.setText("")
        self.label_8.adjustSize()
        self.label_14.setText("")
        self.label_14.adjustSize()
        self.label_9.setText("")
        self.label_9.adjustSize()
        self.label_15.setText("")
        self.label_15.adjustSize()
        self.label_10.setText("")
        self.label_10.adjustSize()
        self.label_16.setText("")
        self.label_16.adjustSize()
        if self.num_bolts == 1:
            self.label_5.setText("Screws 1:")
            self.label_5.adjustSize()
            self.label_12.setText(str(self.positions_bolts[0]))
            self.label_12.adjustSize()
        elif self.num_bolts == 2:
            self.label_5.setText("Screws 1:")
            self.label_5.adjustSize()
            self.label_12.setText(str(self.positions_bolts[0]))
            self.label_12.adjustSize()
            self.label_7.setText("Screws 2:")
            self.label_7.adjustSize()
            self.label_13.setText(str(self.positions_bolts[1]))
            self.label_13.adjustSize()
        elif self.num_bolts == 3:
            self.label_5.setText("Screws 1:")
            self.label_5.adjustSize()
            self.positions_bolts = np.around(self.positions_bolts, 2)
            xx = '[' + str(self.positions_bolts[0][0]) + ', ' + str(self.positions_bolts[0][1]) + ', ' + str(
                self.positions_bolts[0][2]) + ']'
            self.label_12.setText(xx)
            self.label_12.adjustSize()
            self.label_7.setText("Screws 2:")
            self.label_7.adjustSize()
            xx = '[' + str(self.positions_bolts[1][0]) + ', ' + str(self.positions_bolts[1][1]) + ', ' + str(
                self.positions_bolts[1][2]) + ']'
            self.label_13.setText(xx)
            self.label_13.adjustSize()
            self.label_8.setText("Screws 3:")
            self.label_8.adjustSize()
            xx = '[' + str(self.positions_bolts[2][0]) + ', ' + str(self.positions_bolts[2][1]) + ', ' + str(
                self.positions_bolts[2][2]) + ']'
            self.label_14.setText(xx)
            self.label_14.adjustSize()
        elif self.num_bolts == 4:
            self.label_5.setText("Screws 1:")
            self.label_5.adjustSize()
            self.positions_bolts = np.around(self.positions_bolts, 2)
            xx = '[' + str(self.positions_bolts[0][0]) + ', ' + str(self.positions_bolts[0][1]) + ', ' + str(
                self.positions_bolts[0][2]) + ']'
            self.label_12.setText(xx)
            self.label_12.adjustSize()
            self.label_7.setText("Screws 2:")
            self.label_7.adjustSize()
            xx = '[' + str(self.positions_bolts[1][0]) + ', ' + str(self.positions_bolts[1][1]) + ', ' + str(
                self.positions_bolts[1][2]) + ']'
            self.label_13.setText(xx)
            self.label_13.adjustSize()
            self.label_8.setText("Screws 3:")
            self.label_8.adjustSize()
            xx = '[' + str(self.positions_bolts[2][0]) + ', ' + str(self.positions_bolts[2][1]) + ', ' + str(
                self.positions_bolts[2][2]) + ']'
            self.label_14.setText(xx)
            self.label_14.adjustSize()
            self.label_9.setText("Screws 4:")
            self.label_9.adjustSize()
            xx = '[' + str(self.positions_bolts[3][0]) + ', ' + str(self.positions_bolts[3][1]) + ', ' + str(
                self.positions_bolts[3][2]) + ']'
            self.label_15.setText(str(self.positions_bolts[3]))
            self.label_15.adjustSize()

        self.points__ = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        # create the topology of the point
        self.vertices__ = vtk.vtkCellArray()

        # Setup colors
        self.Colors__ = vtk.vtkUnsignedCharArray()
        self.Colors__.SetNumberOfComponents(3)
        self.Colors__.SetName("Colors__")
        for i in range(self.bolts.shape[0]):
            dp = self.bolts[i]
            id = self.points__.InsertNextPoint(dp[0], dp[1], dp[2])
            self.vertices__.InsertNextCell(1)
            self.vertices__.InsertCellPoint(id)
            self.Colors__.InsertNextTuple3(255, 0, 0)

        ## VTK color representation   22222222
        polydata__ = vtk.vtkPolyData()
        polydata__.SetPoints(self.points__)
        polydata__.SetVerts(self.vertices__)
        polydata__.GetPointData().SetScalars(self.Colors__)
        polydata__.Modified()

        glyphFilter__ = vtk.vtkVertexGlyphFilter()
        glyphFilter__.SetInputData(polydata__)
        glyphFilter__.Update()

        dataMapper__ = vtk.vtkPolyDataMapper()
        dataMapper__.SetInputConnection(glyphFilter__.GetOutputPort())

        # Create an actor
        self.actor__ = vtk.vtkActor()
        self.actor__.SetMapper(dataMapper__)

        self.ren__.AddActor(self.actor__)
        self.ren__.ResetCamera()

    def filter_bolts_but_not_to_show(self):
        if self.search_bolts_again != self.fileName:
            self.positions_bolts, self.num_bolts, self.bolts = find_bolts(self.motor_points_forecast_in_robot, eps=3,
                                                                          min_points=50)
        self.search_bolts_again = self.fileName

        self.clear()
        self.label_2.setText("Estimated positions of bolts")
        self.label_2.adjustSize()
        self.label_6.setText("Number of cover screws:")
        self.label_6.adjustSize()
        self.label_11.setText(str(self.num_bolts))
        self.label_11.adjustSize()

        if self.num_bolts == 1:
            self.label_5.setText("Screws 1:")
            self.label_5.adjustSize()
            self.label_12.setText(str(self.positions_bolts[0]))
            self.label_12.adjustSize()
        elif self.num_bolts == 2:
            self.label_5.setText("Screws 1:")
            self.label_5.adjustSize()
            self.label_12.setText(str(self.positions_bolts[0]))
            self.label_12.adjustSize()
            self.label_7.setText("Screws 2:")
            self.label_7.adjustSize()
            self.label_13.setText(str(self.positions_bolts[1]))
            self.label_13.adjustSize()
        elif self.num_bolts == 3:
            self.label_5.setText("Screws 1:")
            self.label_5.adjustSize()
            self.positions_bolts = np.around(self.positions_bolts, 2)
            xx = '[' + str(self.positions_bolts[0][0]) + ', ' + str(self.positions_bolts[0][1]) + ', ' + str(
                self.positions_bolts[0][2]) + ']'
            self.label_12.setText(xx)
            self.label_12.adjustSize()
            self.label_7.setText("Screws 2:")
            self.label_7.adjustSize()
            xx = '[' + str(self.positions_bolts[1][0]) + ', ' + str(self.positions_bolts[1][1]) + ', ' + str(
                self.positions_bolts[1][2]) + ']'
            self.label_13.setText(xx)
            self.label_13.adjustSize()
            self.label_8.setText("Screws 3:")
            self.label_8.adjustSize()
            xx = '[' + str(self.positions_bolts[2][0]) + ', ' + str(self.positions_bolts[2][1]) + ', ' + str(
                self.positions_bolts[2][2]) + ']'
            self.label_14.setText(xx)
            self.label_14.adjustSize()
        elif self.num_bolts == 4:
            self.label_5.setText("Screws 1:")
            self.label_5.adjustSize()
            self.positions_bolts = np.around(self.positions_bolts, 2)
            xx = '[' + str(self.positions_bolts[0][0]) + ', ' + str(self.positions_bolts[0][1]) + ', ' + str(
                self.positions_bolts[0][2]) + ']'
            self.label_12.setText(xx)
            self.label_12.adjustSize()
            self.label_7.setText("Screws 2:")
            self.label_7.adjustSize()
            xx = '[' + str(self.positions_bolts[1][0]) + ', ' + str(self.positions_bolts[1][1]) + ', ' + str(
                self.positions_bolts[1][2]) + ']'
            self.label_13.setText(xx)
            self.label_13.adjustSize()
            self.label_8.setText("Screws 3:")
            self.label_8.adjustSize()
            xx = '[' + str(self.positions_bolts[2][0]) + ', ' + str(self.positions_bolts[2][1]) + ', ' + str(
                self.positions_bolts[2][2]) + ']'
            self.label_14.setText(xx)
            self.label_14.adjustSize()
            self.label_9.setText("Screws 4:")
            self.label_9.adjustSize()
            xx = '[' + str(self.positions_bolts[3][0]) + ', ' + str(self.positions_bolts[3][1]) + ', ' + str(
                self.positions_bolts[3][2]) + ']'
            self.label_15.setText(str(self.positions_bolts[3]))
            self.label_15.adjustSize()
        else:
            self.label_5.setText("Screws 1:")
            self.label_5.adjustSize()
            self.label_12.setText(str(self.positions_bolts[0]))
            self.label_12.adjustSize()
            self.label_7.setText("Screws 2:")
            self.label_7.adjustSize()
            self.label_13.setText(str(self.positions_bolts[1]))
            self.label_13.adjustSize()
            self.label_8.setText("Screws 3:")
            self.label_8.adjustSize()
            self.label_14.setText(str(self.positions_bolts[2]))
            self.label_14.adjustSize()
            self.label_9.setText("Screws 4:")
            self.label_9.adjustSize()
            self.label_15.setText(str(self.positions_bolts[3]))
            self.label_15.adjustSize()
            self.label_10.setText("Screws 5:")
            self.label_10.adjustSize()
            self.label_16.setText(str(self.positions_bolts[4]))
            self.label_16.adjustSize()

    def display_gear(self):
        if self.cover_existence > 0:
            self.clear()
            self.label_6.setText("Warning:")
            self.label_6.adjustSize()
            self.label_11.setText("Cover has not been removed\nno gear point found")
            self.label_11.adjustSize()
            return
        self.ren__.RemoveActor(self.actor__)
        if self.search_bolts_again != self.fileName:
            if self.type <= 2:
                self.gear, self.posgearaup, self.posgearadown = find_geara(
                    seg_motor=self.motor_points_forecast_in_robot)
            else:
                self.gear, self.posgearb = find_gearb(seg_motor=self.motor_points_forecast_in_robot)
        self.search_bolts_again = self.fileName
        self.clear()
        self.num_gear = 2 if self.type <= 2 else 1
        # self.label_2.setText("Info about gears")
        # self.label_2.adjustSize()
        self.label_6.setText("Number of gears:")
        self.label_6.adjustSize()
        self.label_11.setText(str(self.num_gear))
        self.label_11.adjustSize()
        if self.num_gear == 1:
            self.label_5.setText("TypeB gear:")
            self.label_5.adjustSize()
            self.posgearb = np.around(self.posgearb, 2)
            xx = '[' + str(self.posgearb[0]) + ', ' + str(self.posgearb[1]) + ', ' + str(self.posgearb[2]) + ']'
            self.label_12.setText(xx)
            self.label_12.adjustSize()
        elif self.num_gear == 2:
            if self.type == 2:
                self.label_5.setText("TypeA2 upper gear:")
                self.label_5.adjustSize()
                self.posgearaup = np.around(self.posgearaup, 2)
                xx = '[' + str(self.posgearaup[0]) + ', ' + str(self.posgearaup[1]) + ', ' + str(
                    self.posgearaup[2]) + ']'
                self.label_12.setText(xx)
                self.label_12.adjustSize()
                self.label_7.setText("TypeA2 lower gear")
                self.label_7.adjustSize()
                self.posgearadown = np.around(self.posgearadown, 2)
                xx = '[' + str(self.posgearadown[0]) + ', ' + str(self.posgearadown[1]) + ', ' + str(
                    self.posgearadown[2]) + ']'
                self.label_13.setText(xx)
                self.label_13.adjustSize()
            else:
                self.label_5.setText("TypeA1 upper gear:")
                self.label_5.adjustSize()
                self.posgearaup = np.around(self.posgearaup, 2)
                xx = '[' + str(self.posgearaup[0]) + ', ' + str(self.posgearaup[1]) + ', ' + str(
                    self.posgearaup[2]) + ']'
                self.label_12.setText(xx)
                self.label_12.adjustSize()
                self.label_7.setText("TypeA1 upper gear")
                self.label_7.adjustSize()
                self.posgearadown = np.around(self.posgearadown, 2)
                xx = '[' + str(self.posgearadown[0]) + ', ' + str(self.posgearadown[1]) + ', ' + str(
                    self.posgearadown[2]) + ']'
                self.label_13.setText(xx)
                self.label_13.adjustSize()
        # create the geometry of a point
        self.points__ = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        # create the topology of the point
        self.vertices__ = vtk.vtkCellArray()

        # Setup colors
        self.Colors__ = vtk.vtkUnsignedCharArray()
        self.Colors__.SetNumberOfComponents(3)
        self.Colors__.SetName("Colors__")
        for i in range(self.gear.shape[0]):
            dp = self.gear[i]
            id = self.points__.InsertNextPoint(dp[0], dp[1], dp[2])
            self.vertices__.InsertNextCell(1)
            self.vertices__.InsertCellPoint(id)
            if dp[3] == 8:
                r = color_map["upgear_a"][0]
                g = color_map["upgear_a"][1]
                b = color_map["upgear_a"][2]
            elif dp[3] == 7:
                r = color_map["lowgear_a"][0]
                g = color_map["lowgear_a"][1]
                b = color_map["lowgear_a"][2]
            else:
                r = color_map["gear_b"][0]
                g = color_map["gear_b"][1]
                b = color_map["gear_b"][2]
            self.Colors__.InsertNextTuple3(r, g, b)

        ## VTK color representation   22222222
        polydata__ = vtk.vtkPolyData()
        polydata__.SetPoints(self.points__)
        polydata__.SetVerts(self.vertices__)
        polydata__.GetPointData().SetScalars(self.Colors__)
        polydata__.Modified()

        glyphFilter__ = vtk.vtkVertexGlyphFilter()
        glyphFilter__.SetInputData(polydata__)
        glyphFilter__.Update()

        dataMapper__ = vtk.vtkPolyDataMapper()
        dataMapper__.SetInputConnection(glyphFilter__.GetOutputPort())

        # Create an actor
        self.actor__ = vtk.vtkActor()
        self.actor__.SetMapper(dataMapper__)

        self.ren__.AddActor(self.actor__)
        self.ren__.ResetCamera()

    def find_gear_but_not_to_show(self):
        if self.search_bolts_again != self.fileName:
            if self.type <= 2:
                self.gear, self.posgearaup, self.posgearadown = find_geara(
                    seg_motor=self.motor_points_forecast_in_robot)
            else:
                self.gear, self.posgearb = find_gearb(seg_motor=self.motor_points_forecast_in_robot)
        self.search_bolts_again = self.fileName
        self.clear()
        self.num_gear = 2 if self.type <= 2 else 1
        self.label_2.clear()
        # self.label_2.setText("Info about gears")
        # self.label_2.adjustSize()
        self.label_6.setText("Number of gears:")
        self.label_6.adjustSize()
        self.label_11.setText(str(self.num_gear))
        self.label_11.adjustSize()

        if self.num_gear == 1:
            self.label_5.setText("TypeB gear:")
            self.label_5.adjustSize()
            self.posgearb = np.around(self.posgearb, 2)
            xx = '[' + str(self.posgearb[0]) + ', ' + str(self.posgearb[1]) + ', ' + str(self.posgearb[2]) + ']'
            self.label_12.setText(xx)
            self.label_12.adjustSize()
        elif self.num_gear == 2:
            if self.type == 2:
                self.label_5.setText("TypeA2 upper gear:")
                self.label_5.adjustSize()
                self.posgearaup = np.around(self.posgearaup, 2)
                xx = '[' + str(self.posgearaup[0]) + ', ' + str(self.posgearaup[1]) + ', ' + str(
                    self.posgearaup[2]) + ']'
                self.label_12.setText(xx)
                self.label_12.adjustSize()
                self.label_7.setText("TypeA2 lower gear")
                self.label_7.adjustSize()
                self.posgearadown = np.around(self.posgearadown, 2)
                xx = '[' + str(self.posgearadown[0]) + ', ' + str(self.posgearadown[1]) + ', ' + str(
                    self.posgearadown[2]) + ']'
                self.label_13.setText(xx)
                self.label_13.adjustSize()
            else:
                self.label_5.setText("TypeA1 upper gear:")
                self.label_5.adjustSize()
                self.posgearaup = np.around(self.posgearaup, 2)
                xx = '[' + str(self.posgearaup[0]) + ', ' + str(self.posgearaup[1]) + ', ' + str(
                    self.posgearaup[2]) + ']'
                self.label_12.setText(xx)
                self.label_12.adjustSize()
                self.label_7.setText("TypeA1 upper gear")
                self.label_7.adjustSize()
                self.posgearadown = np.around(self.posgearadown, 2)
                xx = '[' + str(self.posgearadown[0]) + ', ' + str(self.posgearadown[1]) + ', ' + str(
                    self.posgearadown[2]) + ']'
                self.label_13.setText(xx)
                self.label_13.adjustSize()

    def save(self):
        sampled = np.asarray(self.motor_points_forecast_in_robot)
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


def find_covers(seg_motor):
    bottom = []
    for point in seg_motor:
        if point[3] == 1: bottom.append(point[0:3])
    bottom = np.array(bottom)
    if bottom.shape[0] < 1000:
        return -1, None, None
    filename = os.getcwd()
    filename = filename + "/cover.pcd"
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
        if point[3] == 9: gearb.append(point[0:4])

    positionb = np.squeeze(np.mean(gearb[0:3], axis=1))

    return np.array(gearb), positionb


def open3d_save_pcd(pc, filename):
    sampled = np.asarray(pc)
    PointCloud_koordinate = sampled[:, 0:3]

    # visuell the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True)


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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Mywindow()
    window.show()
    sys.exit(app.exec_())
