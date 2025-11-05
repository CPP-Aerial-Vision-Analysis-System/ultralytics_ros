#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ultralytics_ros.msg import ImageResult
from mavros_msgs.srv import CommandLong, SetMode, WaypointSetCurrent, WaypointPull
from mavros_msgs.msg import WaypointReached, VfrHud, StatusText, WaypointList
from sensor_msgs.msg import NavSatFix, Image
from std_msgs.msg import Bool
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterEvent

from cv_bridge import CvBridge

from interfaces.srv import GetGPSData, AddWaypoint, DelWaypoint
from wp_sender.parameter import ParameterManager

import time, cv2, math, sys, os, subprocess

ALT = 16.8      # in meters (this is ~55ft)

class Detection_Object:
    def __init__(self, type, confidence, waypoint_index):
        self.type = type          # person or tent
        self.confidence = confidence     
        self.waypoint = waypoint_index       # index > 0

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')

        # Subscribers
        self.create_subscription(ImageResult, "/image_detection", self.image_result_cb, 1)
        self.create_subscription(WaypointList, "/mavros/mission/waypoints", self.waypoints_cb, 1)
        self.create_subscription(WaypointReached, "/mavros/mission/reached", self.update_waypoint_reached, 1)
        self.create_subscription(ParameterEvent, "/parameter_events", self.parameter_event_cb, 10)

        # Publishers

        # Clients
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Set mode service not available, waiting ...")

        # variables
        self.last_before_rtl = 0
        self.next_after_takeoff = 0
        self.takeoff_index = 0
        self.rtl_index = 0
        self.lap = 0
        self.waypoint_reached = 0
        
        self.param_manager = ParameterManager()

        self.fetch_mission_indices()

        self.waypoints = []
        self.detections = {
            "person": Detection_Object(type="person", confidence=0, waypoint_index=0),
            "tent": Detection_Object(type="tent", confidence=0, waypoint_index=0) 
        }

    def fetch_mission_indices(self):
        wp_params = ['num_waypoints', 'takeoff_index', 'rtl_index', 'next_after_takeoff', 'last_before_rtl']
        params = self.param_manager.get_param(self.param_manager.waypoint_client, list_params=wp_params)
        num_waypoints, takeoff_index, rtl_index, next_after_takeoff, last_before_rtl = params.values()
        
        self.num_waypoints = int(num_waypoints)
        self.takeoff_index = int(takeoff_index)
        self.rtl_index = int(rtl_index)
        self.next_after_takeoff = int(next_after_takeoff)
        self.last_before_rtl = int(last_before_rtl)

    def parameter_event_cb(self, msg: ParameterEvent):
        if msg.node == "/waypoint_manager":
            for changed_param in msg.changed_parameters:
                name = changed_param.name
                value = changed_param.value

                if name in ["num_waypoints", "takeoff_index", "rtl_index", "next_after_takeoff", "last_before_rtl"]:
                    self.get_logger().info(f"[Param Update] {name} changed")
                    self.fetch_mission_indices()
                    break

    def update_waypoint_reached(self, msg):
        self.waypoint_reached = msg.wp_seq      # store latest waypoint index   
        self.get_logger().info(f"Current waypoint: {self.waypoint_reached}")

        if self.waypoint_reached == self.last_before_rtl:
            person_lat, person_lon, person_alt = self.get_waypoint(self.detections["person"].waypoint_index)
            tent_lat, tent_lon, tent_alt = self.get_waypoint(self.detections["tent"].waypoint_index)
            self.change_mode("GUIDED")
            self.send_waypoint_data(person_lat, person_lon, person_alt, self.last_before_rtl + 1)
            self.send_waypoint_data(tent_lat, tent_lon, tent_alt, self.last_before_rtl + 2)
            self.change_mode("AUTO")
            # not finished check mission to see
            return
        
        if self.waypoint_reached == self.rtl_index:
            self.get_logger().info("Returning to launch. Mission complete.")

        
    def waypoints_cb(self, msg: WaypointList):
        self.waypoints = msg.waypoints

    def get_waypoint(self, waypoint_index):     # return copy of an old waypoint given index
        if 0 < waypoint_index < len(self.waypoints):
            wp = self.waypoints[waypoint_index]
            lat = wp.x_lat
            lon = wp.y_long
            alt = wp.z_alt
            return lat, lon, alt
        else:
            self.get_logger().warn(f"Waypoint index {waypoint_index} out of range")
            return None

    def image_result_cb(self, msg):
        if msg.detections.detections:
            self.get_logger().info(f"{len(msg.detections.detections)} object(s) detected!")
            
            for detection in msg.detections.detections:
                for result in detection.results:
                    obj_class = result.hypothesis.class_id
                    obj_conf = result.hypothesis.score

                    if obj_class in self.detections:        # only works if obj_class is saved as 'person' or 'tent'    // TODO: DOUBLE CHECK THIS
                        if obj_conf > self.detections[obj_class].confidence:        # get highest conf
                            self.get_logger().info(f"Updating {obj_class}: old_conf={self.detections[obj_class].confidence:.2f}, new_conf={obj_conf:.2f}")
                            # update conf
                            self.detections[obj_class].confidence = obj_conf
                            # update wp_index
                            self.detections[obj_class].waypoint_index = msg.waypoint_index
        else:
            self.get_logger().info_throttle(5, "No objects detected.")

    def change_mode(self, mode):
        # set_mode service should already be ready from self._wait_for_services
        self.get_logger().info(f"Setting mode to {mode}...")
        try:
            req = SetMode.Request()
            req.custom_mode = mode
            future = self.set_mode_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            
            if response.mode_sent:
                self.get_logger().info(f"Mode changed to {mode}")
            else:
                self.get_logger().error("Failed to change mode")
        except Exception as e:
            self.get_logger().error(str(e))

    def send_waypoint_data(self, lat, long, alt, index):
        self.get_logger().info("called waypoint function")
        self.add_wp_client = self.create_client(AddWaypoint, "/addWaypoint")            # need waypoint.py to be running
        while not self.add_wp_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for add waypoint service ...")
        self.get_logger().info("addition service loaded")

        try:
            req = AddWaypoint.Request()
            req.altitude = alt
            req.longitude = long
            req.latitude = lat
            req.index = index
            future = self.add_wp_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            return future.result()      # add .success if we bool otherwise keep if we want Response obj
        except Exception as e:
            self.get_logger().error(str(e))

if __name__ == "__main__":
    rclpy.init()
    node = MainController()
    rclpy.spin(node)    
