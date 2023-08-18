#! /usr/bin/env python3

from ast import Num
from re import X
from tkinter import W
import pyfirmata2
import time
import ROS_Drehtisch_Call
import ROS_Drehtisch
import ROS_Pneumatik_Call
import ROS_Pneumatik
import datetime
from pathlib import Path
import open3d as o3d
import numpy as np
import zivid
from scipy.spatial.transform import Rotation
import urx
import rospy
import moveit_commander
import sys
from util import *
import rosnode
import dynamic_reconfigure.client
from zivid_camera.srv import *
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2

'''
WICHTIG: JOINT LIMITS IN YAML HAVE TO BE CONFIGURED!!!
Nur MoveIT! config .yaml anzupassen führt manchmal trotzdem zu komischen Posen, weil MoveIT! das gekonnt ignoriert :)
paths:
/home/wbk-ur2/dual_ws/src/universal_robot/ur_description/config/ur10e/joint_limits.yaml
/home/wbk-ur2/dual_ws/src/dual_ur10e/dual_ur10e_moveit_config/config/joint_limits.yaml
shoulder_pan joint --> restrict from 60° to 150° (Standard is -360° to 360°), this makes it fail for some poses but stops strange path planning/emergency stops.
REVERT AFTER USING THIS FILE!!!

HOW TO USE:
Change Endeffector in this file (Bottom of the file, "zivid_tool_center_point"):
/home/wbk-ur2/dual_ws/src/dual_ur10e/dual_ur10e_description/urdf/ur10e_zivid_gripper_macro.xacro
TARGET:    <origin xyz="-0.06792999 -0.10424841 0.12798334" rpy="-0.0166622 0.2715963 -0.0058834"/>
Standard:  <!-- <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>  -->

Start Robot: cd dual_ws
roslaunch agiprobot_control bringup_dual.launch type:="real"
Make sure Robot is on External Control

In a new terminal (4 new Terminals + 1 prior Terminal from robot control = 5 Total):
cd /home/wbk-ur2/dual_ws/src/model_builder_inspection/scripts
python3 ROS_Drehtisch.py
python3 ROS_Pneumatik.py
Start ROS_Listener_DatenaufnahmeGlobe.py
Start ROS_Datenaufnahme_Camera_On_Globe.py

Procedure:
ROS_Datenaufnahme_Camera_On_Globe.py asks for confirmation if Motor is clamped in:
--> 1 + Enter --> Program should run without problems, Drehtisch has to initialize and wait 20 secs in case it was stopped at 360° before use

--> 2 + Enter --> Go into Terminal running ROS_Pneumatik.py and follow instructions! --> Program should run without problems, Drehtisch has to initialize and wait 20 secs.

FILES ARE FOUND IN: /home/wbk-ur2/dual_ws/src/model_builder_inspection/scripts/Dataset_Globe
Changes are to be done in the ROS_Listener_DatenaufnahmeGlobe.py (e.g. Change Save-Path, Save different files like rgb.img instead rgb.png etc.)
'''

PORT_PNEU = '/dev/ttyACM0'
# PORT =  pyfirmata2.Arduino.AUTODETECT
board = pyfirmata2.Arduino(PORT_PNEU)
VPPE1 = board.get_pin('d:3:p')
VPPE2 = board.get_pin('d:5:p')
rob = None
robot_ip = '172.22.132.6'

R0 = np.array([[0, -1, 0, 0.62984],
                    [1, 0, 0, 0.637226],
                    [0, 0, 1, 0.214449],
                    [0, 0, 0, 1]])

#set center of rotary table a bit higher so the motor is centered better in the pictures
R0 = np.array([[0, -1, 0, 0.62984],
                    [1, 0, 0, 0.637226],
                    [0, 0, 1, 0.354449],
                    [0, 0, 0, 1]])

# transformsmatrix form world to base coordinate, axis orientation see ros
R1 = np.array([[0, 1, 0, -0.860],
                    [-1, 0, 0, -0.140],
                    [0, 0, 1, 1.1],
                    [0, 0, 0, 1]])

R2 = np.array([[0.96333, 0.00567, 0.26827, -67.92999],
                    [-0.01035, 0.99982, 0.01605, -104.24841],
                    [-0.26813, -0.01824, 0.96321, 127.98334],
                    [0.0, 0.0, 0.0, 1.0]])


def capture():
    # Connect to and call capture assistant service
    print("Obtaining parameters for capture")
    rospy.wait_for_service("/zivid_camera/capture_assistant/suggest_settings", rospy.Duration(5))
    suggest_srv = rospy.ServiceProxy("/zivid_camera/capture_assistant/suggest_settings",
                                     service_class=CaptureAssistantSuggestSettings)
    suggest_req = CaptureAssistantSuggestSettingsRequest()
    suggest_req.ambient_light_frequency = suggest_req.AMBIENT_LIGHT_FREQUENCY_50HZ
    suggest_req.max_capture_time = rospy.Duration(6)
    suggest_srv.call(suggest_req)

    # Capture with suggested settings
    print("Perfoming capture with obtained parameters")
    rospy.wait_for_service("/zivid_camera/capture_assistant/suggest_settings", rospy.Duration(1))
    capture_srv = rospy.ServiceProxy("/zivid_camera/capture", service_class=Capture)
    # capture_req = CaptureRequest()
    capture_srv.call(CaptureRequest())


# this method takes the picture in a random position, facing the center of the rotation table

def _generate_model_dataset(num_captures):
        # with phi limited (sph = [phi, theta, r])

    location_dir = "/home/wbk-ur2/dual_ws/src/model_builder_inspection/scripts/Dataset_Globe/Motor_xxx_" + datetime.datetime.now().strftime(
    "%Y_%m_%d_%H:%M:%S")

    if not Path(location_dir).is_dir():
        Path(location_dir).mkdir(parents=True)

    RGB_dir = location_dir + "/RGB"

    if not Path(RGB_dir).is_dir():
        Path(RGB_dir).mkdir(parents=True)

    depth_dir = location_dir + "/Depth"

    if not Path(depth_dir).is_dir():
        Path(depth_dir).mkdir(parents=True)

    depth_numpy_dir = location_dir + "/Depth_raw_numpy"

    if not Path(depth_numpy_dir).is_dir():
        Path(depth_numpy_dir).mkdir(parents=True)

    file_name_pub.publish(location_dir)

    ROS_Drehtisch_Call.drehen("absolut", 0)

    count_successful_captures = 0

    while count_successful_captures < num_captures:

        mg.stop()
        waypoints = []
        
        radius = 0.6    #distance of camera
        phi = (np.random.random(1)) * 3 * np.pi / 8 + np.pi/4   #horizontal restrictions
        theta = np.random.random(1) * 3 * np.pi / 8 + np.pi/8   #vertical restrictions

        pos = np.asarray([radius*np.sin(theta)*np.cos(phi), radius*np.sin(theta)*np.sin(phi), radius*np.cos(theta)])
        alpha, beta, gamma = get_cam_rotation(pos)
        rotation = Rotation.from_euler("XYZ", [alpha, beta, np.pi], degrees=False)
        camera_pose_in_teller = np.eye(4)
        camera_pose_in_teller[0, -1] = pos[0]
        camera_pose_in_teller[1, -1] = pos[1]
        camera_pose_in_teller[2, -1] = pos[2]
        camera_pose_in_teller[:3, :3] = rotation.as_matrix()
        #print("camera pose in teller KOS", camera_pose_in_teller)

        calculated_eef_pose_in_world = R1@R0@camera_pose_in_teller
        r = Rotation.from_matrix(calculated_eef_pose_in_world[:3, :3])
        orientation = r.as_quat()
        #print("camera pose in world KOS", calculated_eef_pose_in_world)

        target_pose = PoseStamped()
        target_pose.header.frame_id = "world"
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = calculated_eef_pose_in_world[0, -1]
        target_pose.pose.position.y = calculated_eef_pose_in_world[1, -1]
        target_pose.pose.position.z = calculated_eef_pose_in_world[2, -1]
        target_pose.pose.orientation.x = orientation[0]
        target_pose.pose.orientation.y = orientation[1]
        target_pose.pose.orientation.z = orientation[2]
        target_pose.pose.orientation.w = orientation[3]

        waypoints.append(target_pose.pose)
        fraction = 0.0
        maxtries = 20
        attempts = 0

        #mg.set_start_state_to_current_state()

        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = mg.compute_cartesian_path(waypoints, 0.03, 0.0, True)
            attempts += 1
            if attempts % 10 == 0:
                print(" attempts:", attempts, "fraction:", fraction)

        if fraction == 1.0:
            success = mg.execute(plan)
            target_angle = np.random.randint(0, 360)
            ROS_Drehtisch_Call.drehen("absolut", target_angle)
            if success:
                capture()
                count_successful_captures += 1
                success = rospy.wait_for_message('PC_Handler_Update', String)

        else:
            print("Error, Solver cant find solution")


# moves roboter to given position
def move_to_pos(position):
    mg.set_start_state_to_current_state()
    mg.set_joint_value_target(position)

    result = mg.plan()
    if result[0] == False:
        rospy.logerr("Failed to plan!")
        sys.exit(0)
        # Execute planning result
    if not mg.execute(result[1]):
        print("Trajectory execution failed")
        exit(1)
    mg.stop()


def set_robot_pose(orientation):
    target_pose = PoseStamped()
    target_pose.pose.orientation.x = orientation[0]
    target_pose.pose.orientation.y = orientation[1]
    target_pose.pose.orientation.z = orientation[2]
    target_pose.pose.orientation.w = orientation[3]
    mg.set_pose_target(target_pose)


if __name__ == "__main__":

    # Initalize ROS and moveit
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("Datenaufnahme")
    file_name_pub = rospy.Publisher('file_name', String, queue_size=10) 
    # mg = moveit_commander.MoveGroupCommander("zivid_ur10e_arm")
    # mg.set_planner_id("PersistentPRMstar")
    # mg.allow_replanning(True)


    userinput = (input(
        "Ist der Motor bereits eingespannt und die Pneumatikansteuerung kann übersprungen werden?: \n Ja = 1 \n Nein die Pneumatik muss geöffnet werden um Motor einzuspannen = 2 \n"))

    if userinput == "1":
        #_generate_model_dataset(10) #SPECIFY NUM_CAPTURES
        ROS_Pneumatik_Call.pneumatik("end")
    elif userinput == "2":
        ROS_Pneumatik_Call.pneumatik("end")
        ROS_Pneumatik_Call.pneumatik("open")
        
        ROS_Pneumatik_Call.pneumatik("close")
        #_generate_model_dataset(10) #SPECIFY NUM_CAPTURES
        ROS_Pneumatik_Call.pneumatik("end")
    else:
        print("Unzulässige Eingabe")

