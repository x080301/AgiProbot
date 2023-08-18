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


def capture_one_motor():
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
    zivid_position = deg_to_rad([91.2, -123.65, -41.74, -131.42, 57.83, 129.86])
    # zivid_position = deg_to_rad([86.98, -110.92, -74.07, -106.37, 50.51, 127.13])
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

    while True
        userinput = input(
            "Ist der Motor bereits eingespannt und die Pneumatikansteuerung kann ¨¹bersprungen werden?: \n Ja = 1 \n Nein die Pneumatik muss ge?ffnet werden um Motor einzuspannen = andere Tasten \n")

        if userinput == "1":
            capture_one_motor()  # _generate_model_dataset(10) #SPECIFY NUM_CAPTURES
        else:
            ROS_Pneumatik_Call.pneumatik("end")
            ROS_Pneumatik_Call.pneumatik("open")
            ROS_Pneumatik_Call.pneumatik("close")

            capture_one_motor()  # _generate_model_dataset(10) #SPECIFY NUM_CAPTURES
