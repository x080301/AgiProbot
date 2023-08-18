#! /usr/bin/env python3

import sys
import rospy
import numpy as np
from model_builder_inspection.srv import drehen_srv, drehen_srvResponse

'''Call Funktion für Drehtisch. Wurde geschrieben, damit Funktion drehen nicht in jedem Programm definiert werden
muss sondern nur importiert wird.'''

def drehen(operation, x):
    rospy.wait_for_service('drehen_service')
    successful = False
    try:
        drehen_service = rospy.ServiceProxy('drehen_service', drehen_srv)
        resp = drehen_service(operation, x)
        successful = True
        print(resp.response)
        return successful
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return successful

if __name__ == "__main__":
    drehwinkel = 0
    print("Ich stehe gerade auf %a Grad und drehe mich auf %s Grad" % (rospy.get_param("/abs_angle"), drehwinkel))
    drehen("absolut", drehwinkel)
    #print("Habe um %s gedreht" % drehwinkel)
    #print("Ich stehe gerade auf %a Grad und drehe mich um %s Grad" % (rospy.get_param("/abs_angle"), -drehwinkel))
    #drehen("relativ", -drehwinkel)
    #print("Ich stehe gerade auf %a Grad und resette mich auf 0 Grad")
    #drehen("absolut", 0)