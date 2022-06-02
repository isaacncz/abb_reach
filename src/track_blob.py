#!/usr/bin/env python

# https://automaticaddison.com/how-to-convert-camera-pixels-to-robot-base-frame-coordinates/



import rospy
from std_msgs.msg import Int32 # Messages used in the node must be imported.
from geometry_msgs.msg import Pose


import sys
import cv2
import numpy as np


rospy.init_node("track_blob")



pub = rospy.Publisher('follow_blob', Pose, queue_size=10)

target_pose=Pose() # declaring a message variable of type Int32


CM_TO_PIXEL = 60 / 640

# def openCamera():
# 	cap=cv2.VideoCapture(0)
# 	return cap

def remapValue(OldMax,OldMin,NewMax,NewMin,OldValue):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue



def main():
	x2_remap=0.0
	y2_remap=0.0
	x_d_p=0.0
	y_d_p=0.0
	cap=cv2.VideoCapture(0)
	width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )   # float `width`
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
	print(width, height)

	while(1):
		_, img = cap.read()
		if not _:
			break
		#converting frame(img i.e BGR) to HSV (hue-saturation-value)

		hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		blue_lower=np.array([132,100,52],np.uint8)
		blue_upper=np.array([222,222,222],np.uint8)


		blue=cv2.inRange(hsv,blue_lower,blue_upper)
		
		#Morphological transformation, Dilation  	
		# kernal = np.ones((5 ,5), "uint8")


		# blue=cv2.dilate(blue,kernal)

		# img=cv2.circle(img,(260,68),5,(255,0,0),-1)

				
		#Tracking the Blue Color
		contours,hierarchy=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		if len(contours)>0:
			contour= max(contours,key=cv2.contourArea)
			area = cv2.contourArea(contour)
			if area>40: 
				x,y,w,h = cv2.boundingRect(contour)	
				img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

				# Draw circle in the center of the bounding box
				x2 = x + int(w/2)
				y2 = y + int(h/2)
				cv2.circle(img,(x2,y2),4,(0,255,0),-1)

				# Calculate the center of the bounding box in centimeter coordinates
				# instead of pixel coordinates
				x2_cm = x2 * CM_TO_PIXEL
				y2_cm = y2 * CM_TO_PIXEL
				# x2_remap = remapValue(0,60,-0.35,0.35,x2_cm)
				# y2_remap = remapValue(0,44,0.3,0.4,y2_cm)
				# # x2_remap = remapValue(0,32,0.35,0.25,x2_cm)
				# # y2_remap = remapValue(0,23.8,11.9,-11.9,y2_cm)
				# x2_cm = round(x2_cm,7)
				# y2_cm=round(y2_cm,7)
				# x2_remap=round(x2_remap,7)
				# y2_remap=round(y2_remap,7)

				s = "x: " + str(x2_cm) + ", y: " + str(y2_cm)
				top = y - 15 if y - 15 > 15 else y + 15
				cv2.putText(img,s,(x-20,top),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1,cv2.LINE_AA)
			
				if (abs(x2_cm-x_d_p)> 1 or abs(y2_cm-y_d_p)>1):
					target_pose.position.x=x2_cm
					target_pose.position.y=y2_cm
					target_pose.position.z=0.0
					pub.publish(target_pose)
					x_d_p=x2_cm
					y_d_p=y2_cm
				
			
		
		cv2.imshow("Mask",blue)
		cv2.imshow("Color Tracking",img)
		if cv2.waitKey(1)== ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()