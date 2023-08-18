#! /usr/bin/env python3

import sys
import rospy
from model_builder_inspection.srv import pneumatik_srv, pneumatik_srvResponse

'''Call Funktion für Pneumatik Service. Wurde geschrieben, damit Funktion pneumatik nicht in jedem Programm 
definiert werden muss sondern nur importiert wird.'''

def pneumatik(operation):
    rospy.wait_for_service('pneumatik_service')
    successful = False
    try:
        pneumatik_service = rospy.ServiceProxy('pneumatik_service', pneumatik_srv)
        resp = pneumatik_service(operation)
        successful = True
        print(resp.response)
        return successful
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return successful


if __name__ == "__main__":
    # print("Motor ist %s" % (rospy.get_param("/pneumatik_flag")))
    #pneumatik("close")
    #print("Motor eingespannt")
    pneumatik("open")
    print("Motoreinspannung wird gelöst")
