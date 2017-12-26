#!/usr/bin/env python

"""
    object_follower.py - Version 1.1 2013-12-20
    
    Follow a target published on the /roi topic using depth from the depth image.
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2013 Patrick Goebel.  All rights reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
"""

import rospy
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from geometry_msgs.msg import Twist
from math import copysign, isnan
import cv2
import math
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import thread

class ObjectFollower():
    def __init__(self):
        rospy.init_node("object_follower")
                        
        # Set the shutdown function (stop the robot)
        rospy.on_shutdown(self.shutdown)
        
        # How often should we update the robot's motion?
        self.rate = rospy.get_param("~rate", 15)
        
        r = rospy.Rate(self.rate)
        
        # Scale the ROI by this factor to avoid background distance values around the edges
        self.scale_roi = rospy.get_param("~scale_roi", 0.9)
        
        # The max linear speed in meters per second
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.3)
        
        # The minimum linear speed in meters per second
        self.min_linear_speed = rospy.get_param("~min_linear_speed", 0.02) 
        
        # The maximum rotation speed in radians per second
        self.max_rotation_speed = rospy.get_param("~max_rotation_speed", 1.0)
        
        # The minimum rotation speed in radians per second
        self.min_rotation_speed = rospy.get_param("~min_rotation_speed", 0.5)
        
        # The x threshold (% of image width) indicates how far off-center
        # the ROI needs to be in the x-direction before we react
        self.x_threshold = rospy.get_param("~x_threshold", 0.1)
        
        # How far away from the goal distance (in meters) before the robot reacts
        self.z_threshold = rospy.get_param("~z_threshold", 0.05)
        
        # The maximum distance a target can be from the robot for us to track
        self.max_z = rospy.get_param("~max_z", 1.6)

        # The minimum distance to respond to
        self.min_z = rospy.get_param("~min_z", 0.35)
        
        # The goal distance (in meters) to keep between the robot and the person
        self.goal_z = rospy.get_param("~goal_z", 0.6)

        # How much do we weight the goal distance (z) when making a movement
        self.z_scale = rospy.get_param("~z_scale", 0.5)

        # How much do we weight (left/right) of the person when making a movement        
        self.x_scale = rospy.get_param("~x_scale", 0.3)
        
        # Slow down factor when stopping
        self.slow_down_factor = rospy.get_param("~slow_down_factor", 0.3)

        self.CameraHeight = rospy.get_param("~CameraHeight", 215.0)
        self.O3M = rospy.get_param("~O3M", 350.0)
        self.alfax = rospy.get_param("~alfax", 626.1817)
        self.alfay = rospy.get_param("~alfay", 625.9736)
        self.focalpointx = rospy.get_param("~focalpointx", 334.9849)
        self.focalpointy = rospy.get_param("~focalpointy", 242.9379)

        self.alfa_atanValue = 0.0
        self.roi_point_x = 0.0
        self.roi_point_y = 0.0
        self.O1P1 = 0.0
        self.gama_atanValue = 0.0
        self.beta_angle = 0.0
        self.temp_O2P1 = 0.0
        self.O2P = 0.0
        self.PQ = 0.0
        self.O3P = 0.0
        self.real_distance = 0.0

        # Initialize the global ROI
        self.roi = RegionOfInterest()

        # Publisher to control the robot's movement
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        
        # Intialize the movement command
        self.move_cmd = Twist()
        
        # Get a lock for updating the self.move_cmd values
        self.lock = thread.allocate_lock()
        
        # We will get the image width and height from the camera_info topic
        self.image_width = 640
        self.image_height = 480
        
        # We need cv_bridge to convert the ROS depth image to an OpenCV array
        self.cv_bridge = CvBridge()
        self.depth_array = None
        
        # Set flag to indicate when the ROI stops updating
        self.target_visible = False

        rospy.Subscriber('roi', RegionOfInterest, self.set_cmd_vel, queue_size=3)
        
        # Wait until we have an ROI to follow
        rospy.loginfo("Waiting for an ROI to track...")
        
        rospy.wait_for_message('roi', RegionOfInterest)
        
        rospy.loginfo("ROI messages detected. Starting follower...")
        
        # Begin the tracking loop
        while not rospy.is_shutdown():
            # Acquire a lock while we're setting the robot speeds
            self.lock.acquire()
            
            try:
                if not self.target_visible:
                    # If the target is not visible, stop the robot smoothly
                    self.move_cmd.linear.x *= self.slow_down_factor
                    self.move_cmd.angular.z *= self.slow_down_factor
                else:
                    # Reset the flag to False by default
                    self.target_visible = False
                    
                # Send the Twist command to the robot
                self.cmd_vel_pub.publish(self.move_cmd)
                    
            finally:
                # Release the lock
                self.lock.release()
            
            # Sleep for 1/self.rate seconds
            r.sleep()
                        
    def set_cmd_vel(self, msg):
        # Acquire a lock while we're setting the robot speeds
        self.lock.acquire()
        
        try:
            # If the ROI has a width or height of 0, we have lost the target
            if msg.width == 0 or msg.height == 0:
                self.target_visible = False
                return
            else:
                self.target_visible = True
    
            self.roi = msg
                        
            # Compute the displacement of the ROI from the center of the image
            target_offset_x = self.roi.x_offset + self.roi.width / 2 - self.image_width / 2
    
            try:
                percent_offset_x = float(target_offset_x) / (float(self.image_width) / 2.0)
            except:
                percent_offset_x = 0
                                            
            # Rotate the robot only if the displacement of the target exceeds the threshold
            if abs(percent_offset_x) > self.x_threshold:
                # Set the rotation speed proportional to the displacement of the target
                speed = percent_offset_x * self.x_scale
                self.move_cmd.angular.z = -copysign(max(self.min_rotation_speed,
                                            min(self.max_rotation_speed, abs(speed))), speed)
            else:
                self.move_cmd.angular.z = 0
                
            # Now compute the depth component


            self.alfa_atanValue = math.atan(self.CameraHeight / self.O3M)

            self.roi_point_x = self.roi.x_offset + self.roi.width / 2

            self.roi_point_y = self.roi.y_offset + self.roi.height

            self.O1P1 = abs(self.roi_point_y - self.focalpointy)

            self.gama_atanValue = math.atan(self.O1P1 / self.alfay)

            self.beta_angle = self.alfa_atanValue - self.gama_atanValue

            self.temp_O2P1 = math.sqrt(self.O1P1 * self.O1P1 + math.pow(self.alfay, 2))

            self.O2P = self.CameraHeight / math.sin(self.beta_angle)

            self.PQ = self.O2P * abs(self.roi_point_x - self.focalpointx) * self.alfay / (self.alfax * self.temp_O2P1)

            self.O3P = self.CameraHeight / abs(math.tan(self.beta_angle))

            self.real_distance = math.sqrt(self.O3P * self.O3P + self.PQ * self.PQ) / 1000

            # Stop the robot's forward/backward motion by default
            linear_x = 0
            print self.real_distance

            if self.real_distance > self.min_z:
                #print self.real_distance
                    # Check the max range and goal threshold
                if self.real_distance < self.max_z and (abs(self.real_distance - self.goal_z) > self.z_threshold):
                    speed = (self.real_distance - self.goal_z) * self.z_scale
                    linear_x = copysign(min(self.max_linear_speed, max(self.min_linear_speed, abs(speed))), speed)
    
            if linear_x == 0 and self.move_cmd.linear.x > self.min_linear_speed:
                # Stop the robot smoothly
                self.move_cmd.linear.x *= self.slow_down_factor
            else:
                self.move_cmd.linear.x = linear_x
        
        finally:
            # Release the lock
            self.lock.release()

    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        # Unregister the subscriber to stop cmd_vel publishing
        #self.depth_subscriber.unregister()
        rospy.sleep(1)
        # Send an emtpy Twist message to stop the robot
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)      

if __name__ == '__main__':
    try:
        ObjectFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Object follower node terminated.")

