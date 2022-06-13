#! /usr/bin/env python
import rospy
from irb120_env import AbbEnv

from sb3_contrib import TQC

import numpy as np

import rospy
from geometry_msgs.msg import Pose
target_pose=Pose() # declaring a message variable of type Int32

def remapValue(OldMax,OldMin,NewMax,NewMin,OldValue):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

def callback(data):
    global globalPosition
    pos = data.position

    x2_remap = remapValue(0,60,-0.50,0.50,pos.y) #remap value to rviz coordinate, x and y interchange
    y2_remap = remapValue(0,44,0.2,0.4,pos.x)    #remap value to rviz coordinate, x and y interchange

    x2_remap=round(x2_remap,7)
    y2_remap=round(y2_remap,7)

    # print(x2_remap,y2_remap)
    abb.desired_goal = np.array([x2_remap,y2_remap,0.25]) #fix the z axis
    obs = abb._get_obs()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = abb.step(action)

    
def listener():

    rospy.Subscriber("follow_blob", Pose, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':

    model = TQC.load("/home/isaac/Desktop/model_storage/TQC_May-30:11:19:04.zip")

    abb = AbbEnv()

    obs = abb.reset()
    listener()
