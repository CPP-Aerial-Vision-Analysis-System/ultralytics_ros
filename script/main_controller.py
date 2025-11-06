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
        self.waypoint_index = waypoint_index       # index > 0

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
        self.set_mode = self.create_client(SetMode, "/mavros/set_mode")
        while not self.set_mode.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Set mode service not available, waiting ...")

        # variables
        self.last_before_rtl = 0
        self.next_after_takeoff = 0
        self.takeoff_index = 0
        self.rtl_index = 0
        self.lap = 0
        self.waypoint_reached = 0
        self.objects = 0
        
        self.param_manager = ParameterManager()

        self.fetch_mission_indices()

        self.waypoints = []
        self.detections = {
            "person": Detection_Object(type="person", confidence=0, waypoint_index=0),
            "tent": Detection_Object(type="tent", confidence=0, waypoint_index=0) 
        }

        self.add_wp_client = self.create_client(AddWaypoint, '/addWaypoint')
        while not self.add_wp_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for add waypoint service ...')
        
        self.set_current = self.create_client(WaypointSetCurrent, "/mavros/mission/set_current")
        while not self.set_current.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Set current service not available, waiting ...")

    def set_current_waypoint(self, index):
        req = WaypointSetCurrent.Request()
        req.wp_seq = index
        future = self.set_current.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result().success:
            self.get_logger().info(f"Set current mission index to {index}")
        else:
            self.get_logger().warn("Failed to set current waypoint.")
        
        #self.send_waypoint_data(1.313213,1.3190,13123,2)
        

    def fetch_mission_indices(self):
        wp_params = ['num_waypoints', 'takeoff_index', 'rtl_index', 'next_after_takeoff', 'last_before_rtl']
        params = self.param_manager.get_param(self.param_manager.waypoint_client, list_params=wp_params)
        num_waypoints, takeoff_index, rtl_index, next_after_takeoff, last_before_rtl = params.values()
        
        self.num_waypoints = int(num_waypoints)
        self.takeoff_index = int(takeoff_index)
        self.rtl_index = int(rtl_index)
        self.next_after_takeoff = int(next_after_takeoff)
        self.last_before_rtl = int(last_before_rtl)
        self.get_logger().info(f"Mission Indices - Num Waypoints: {self.num_waypoints}, Takeoff Index: {self.takeoff_index}, Next After Takeoff: {self.next_after_takeoff}, Last Before RTL: {self.last_before_rtl}, RTL Index: {self.rtl_index}")

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


        if self.waypoint_reached == self.last_before_rtl and (self.valid_detection("person") and self.valid_detection("tent")):
            # self.send_waypoint_data(-35, -150, 10, self.last_before_rtl + 1)
            # self.send_waypoint_data(-30, -120, 10, self.last_before_rtl + 1)
            self.send_waypoint_data([
                {"lat": -40, "lon": -130, "alt": 10, "index": self.last_before_rtl + 1},
                {"lat": -30, "lon": -120, "alt": 10, "index": self.last_before_rtl + 2}
            ])
        elif self.waypoint_reached == self.last_before_rtl and (self.valid_detection("person") or self.valid_detection("tent")):
            # switch to GUIDED, wait briefly, send waypoint, then return to AUTO
            self.send_waypoint_data([
                {"latitude": -35, "longitude": -125, "altitude": 10, "index": self.last_before_rtl + 1}
            ])
            #self.set_current_waypoint(self.last_before_rtl + 1)
        
        # if self.waypoint_reached == self.last_before_rtl and self.objects == 1:
        #     # drop payload sequence for first object
        #     # if there is a second object, then add it else switch to RTL
        #     self.send_waypoint_data(-35.3631204, 149.1651884, 10, self.last_before_rtl + 2)
            
        #     if self.detections["person"].confidence > 0 and self.detections["tent"].confidence > 0:
        #         person_lat, person_lon, person_alt = self.get_waypoint(self.detections["person"].waypoint_index)
        #         tent_lat, tent_lon, tent_alt = self.get_waypoint(self.detections["tent"].waypoint_index)
        #         self.change_mode("GUIDED")
        #         self.send_waypoint_data(person_lat, person_lon, person_alt, self.last_before_rtl + 1)
        #         self.send_waypoint_data(tent_lat, tent_lon, tent_alt, self.last_before_rtl + 2)
        #         self.change_mode("AUTO")

        #     if self.detections["person"].confidence > 0:
        #         person_lat, person_lon, person_alt = self.get_waypoint(self.detections["person"].waypoint_index)
        #         self.change_mode("GUIDED")
        #         self.send_waypoint_data(person_lat, person_lon, person_alt, self.last_before_rtl + 1)
        #         self.change_mode("AUTO")

        #     if self.detections["tent"].confidence > 0:
        #         tent_lat, tent_lon, tent_alt = self.get_waypoint(self.detections["tent"].waypoint_index)
        #         self.change_mode("GUIDED")
        #         self.send_waypoint_data(tent_lat, tent_lon, tent_alt, self.last_before_rtl + 1)
        #         self.change_mode("AUTO")
        #     # not finished check mission to see
            return
        
        if self.waypoint_reached == self.rtl_index:
            self.get_logger().info("Returning to launch. Mission complete.")

    def valid_detection(self, type):
        if type in self.detections:
            if self.detections[type].confidence > 0:
                return True
        return False

        
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
        """Change flight mode of the drone."""
        self.get_logger().info(f"Chaning mode to {mode}")
        try:
            req = SetMode.Request()
            req.custom_mode = mode
            future = self.set_mode.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()
            if resp is not None and getattr(resp, 'mode_sent', False):
                self.get_logger().info(f"Mode changed to {mode} successfully.")
            else:
                self.get_logger().error(f"Failed to change mode to {mode}")
        except Exception as e:
            self.get_logger().error(f"Exception while changing mode: {e}")

    def send_waypoint_data(self, wp_list):
        for
        self.get_logger().info("called waypoint function")
        req = AddWaypoint.Request()
        req.latitude = float(lat)
        req.longitude = float(long)
        req.altitude = float(alt)
        req.index = index

        future = self.add_wp_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)  # blocks this node until done
        if future.result() is not None:
            resp = future.result()
            self.get_logger().info(f"Service replied: success={resp.success}")
        else:
            self.get_logger().error(f"Service call failed: {future.exception()}")

if __name__ == "__main__":
    rclpy.init()
    node = MainController()
    rclpy.spin(node)    