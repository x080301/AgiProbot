#! /usr/bin/python3
import rospy
import sys
import moveit_commander

from math import pi
import time
from zivid_camera.srv import *
import ROS_Drehtisch_Call

import ROS_Pneumatik_Call


def capture():
    # Connect to and call capture assistant service
    print("Obtaining parameters for capture")
    rospy.wait_for_service("/zivid_camera/capture_assistant/suggest_settings", rospy.Duration(5))
    suggest_srv = rospy.ServiceProxy("/zivid_camera/capture_assistant/suggest_settings",
                                     service_class=CaptureAssistantSuggestSettings)
    suggest_req = CaptureAssistantSuggestSettingsRequest()
    suggest_req.ambient_light_frequency = suggest_req.AMBIENT_LIGHT_FREQUENCY_50HZ
    suggest_req.max_capture_time = rospy.Duration(10)
    suggest_srv.call(suggest_req)

    # Capture with suggested settings
    print("Perfoming capture with obtained parameters")
    rospy.wait_for_service("/zivid_camera/capture_assistant/suggest_settings", rospy.Duration(1))
    capture_srv = rospy.ServiceProxy("/zivid_camera/capture", service_class=Capture)
    # capture_req = CaptureRequest()
    capture_srv.call(CaptureRequest())


def deg_to_rad(list):
    return [e * 2 * pi / 360 for e in list]


# given position (in terms of all robot joints) plan and execute
def move_to_pos(mg, position):
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


def capture_one_motor(view_mode="top"):
    # Initialize ROS and MoveIt!
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("SFB_Demo")
    ROS_Drehtisch_Call.drehen("absolut", 0)

    # bool for stoppping infinite loop
    rospy.set_param('stop_infinite', False)

    # movegroup definitions for dual robots
    # tscan_movegroup = moveit_commander.MoveGroupCommander("tscan_ur10e_arm")
    # tscan_movegroup.set_planner_id("PRMstar")
    zivid_movegroup = moveit_commander.MoveGroupCommander("zivid_ur10e_arm")
    zivid_movegroup.set_planner_id("PRMstar")
    gripper_movegroup = moveit_commander.MoveGroupCommander("gripper")

    # define positions
    if view_mode == "top":
        # zivid_position = deg_to_rad([94.51, -98.09, -98.32, -97.46, 44.5, 122.05])  # top X8
        # zivid_position = deg_to_rad([97.40, -103.11, -92.68, -102.92, 42.49, 128.20])
        zivid_position = deg_to_rad([97.40, -102.11, -88.97, -107.92, 42.56, 128.32])
    else:
        zivid_position = deg_to_rad([90.40, -92.77, -95.07, -104.55, 50.08, 27.16])  # bottom X 8

    rotation_per_capture = 45

    # while True:
    ROS_Drehtisch_Call.drehen("absolut", 0)
    ################################################################################
    # first position Zivid (capture, rotate table, capture again)

    clamping_system_angle = 0
    while clamping_system_angle < 360:
        print(clamping_system_angle)

        if clamping_system_angle == 0:
            move_to_pos(zivid_movegroup, zivid_position)
        else:
            ROS_Drehtisch_Call.drehen("relativ", rotation_per_capture)

        capture()
        time.sleep(1)
        clamping_system_angle += rotation_per_capture


if __name__ == "__main__":

    while True:
        while True:
            userinput = input(
                "Ist der Motor bereits eingespannt und die Pneumatikansteuerung kann Übersprungen werden?: \n Ja = 1 \n Nein die Pneumatik muss geöffnet werden um Motor einzuspannen = andere Tasten \n")

            if userinput == "1":
                break
            else:
                ROS_Pneumatik_Call.pneumatik("end")
                ROS_Pneumatik_Call.pneumatik("open")
                ROS_Pneumatik_Call.pneumatik("close")

        while True:
            userinput = input("top view = 1, bottom view = 2")
            if userinput == "1":
                capture_one_motor(view_mode="top")  # _generate_model_dataset(10) #SPECIFY NUM_CAPTURES
                break
            elif userinput == "2":
                capture_one_motor(view_mode="bottom")
                break
            else:
                print("unexpected input! \n top view = 1, bottom view = 2\n")
