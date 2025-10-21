#!/usr/bin/env pythom3
import rclpy
from rclpy.node import Node
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2D
from mavros_msgs.srv import CommandLong, SetMode
from mavros_msgs.msg import WaypointReached, VfrHud, StatusText, State
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import NavSatFix
from rcl_interfaces.srv import GetParameters

from interfaces.srv import GetGPSData, AddWaypoint, DelWaypoint
from wp_sender.wp_sender.parameter import ParameterManager

from collections import deque
import time, cv2, math, sys, os, subprocess


# from  camera_frame import WaypointManager
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

ALT = 16.8  # in meters (this is ~55 ft)

class Detected_Object_Waypoints(Node):
    def __init__(self):
        super().__init__('detected_object_waypoints')
        self.detected_objects = (
            deuque()
        ) # queue to store detected objects + GPS waypoints

    def add_object(self, object_name, lat, long, alt, index):
        """
        Adds a detected object with its name and calculated GPS coordinates to the queue.
        """
        if len(self.detected_objects) >= 4:
            return

        detected_object = {
            "name": object_name,
            "latitude": lat,
            "longitude": long,
            "alt": alt,
            "index": index,
        }
        self.detected_objects.append(detected_object)
        self.get_logger().info(f"Object '{object_name}' added at LAT: {lat}, LONG: {long}, ALT: {alt}")

    def get_detected_objects(self):
        """
        Returns the queue of detected objects with their GPS waypoints.
        """
        return self.detected_objects

    def rotate_waypoints(self, rotate=-1):
        if self.detected_objects:
            self.detected_objects.rotate(rotate)

class YoloResultSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_result_subscriber')

        self.create_subscription(YoloResult, 'yolo_result', self.yolo_result_cb, 1)
        self.create_subscription(WaypointReached, "/mavros/mission/reached", self.update_waypoint_reached, 1)   # original doesn't have queue
        self.create_subscription(VfrHud, "/mavros/vfr_hub", self.speed_cb)
        self.status_pub = self.create_publisher(StatusText, "/mavros/statustext/send", queue=10)

        # Might need to convert this (but original is commmented out)
        # rospy.wait_for_service("/mavros/cmd/command")
        # self.command_service = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)
        # self.waypoint_manager= WaypointManager()

        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")

        # variables
        self.lasttime = time.time()
        self.run_detection_once = False
        self.waypoint_reached = 0
        self.detected_object_waypoints = Detected_Object_Waypoints()
        self.lap = 0

        self.param_manager = ParameterManager()
        class_names = self.param_manager.get_param(self.param_manager.tracknode_client, list_params=['yolo_class_names'], string_value=True)
        num_waypoints = self.param_manager.get_param(self.param_manager.waypoint_client, list_params=['num_waypoints'], integer_value=True)

        while class_names is None:
            self.get_logger().warn(f"Waiting for /yolo_class_names to be set...")
            class_names = self.param_manager.get_param(self.param_manager.tracknode_client, list_params=['yolo_class_names'], string_value=True)
        self.class_names = eval(class_names)

        while num_waypoints is None:
            self.get_logger().warn(f"Waiting for /num_waypoints to be set...")
            num_waypoints = self.param_manager.get_param(self.param_manager.waypoint_client, list_params=['num_waypoints'], integer_value=True)
        self.num_waypoints = int(num_waypoints)

        self.within_geofence = False

        min_lat = min(
            0, 0, 0, 0
        )
        max_lat = max(
            0, 0, 0, 0            
        )

        min_lon = min(
            0, 0, 0, 0
        )
        max_lon = max(
            0, 0, 0, 0
        )

        self.GEOFENCE = {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            # "alt": alt  
        }

        self.create_subscription(NavSatFix, "/mavros/global_position/global", self.geofence_check)

    def geofence_check(self, msg):
        lat = msg.latitude
        long = msg.longitude
        self.within_geofence = (
            self.GEOFENCE["min_lat"] <= lat <= self.GEOFENCE["max_lat"]
            and self.GEOFENCE["min_lon"] <= long <= self.GEOFENCE["max_lon"]
        )

        if self.within_geofence:
            print(f"{GREEN}Geofence status: Inside{RESET}")     # fix
        else:
            print(f"{YELLOW}Geofence status: Outside{RESET}")   # fix

    def send_status(self, text, throttle=False):
        now = time.time()

        if not throttle or (now - self.last_status_time > self.status_interval):
            status_msg = StatusText()
            status_msg.severity = 6  # 6 = NOTICE
            status_msg.text = text
            self.status_pub.publish(status_msg)
            self.last_status_time = now

    def compare_object_names(self, object_name):
        q = self.detected_object_waypoints.get_detected_objects()
        return any(item["name"] == object_name for item in q)

    def set_servo(self, channel, pwm_value):
        try:
            command = CommandLong.Request()
            command.command = 183  # MAV_CMD_DO_SET_SERVO
            command.param1 = channel
            command.param2 = pwm_value
            response = self.command_service(command)            # maybe error here
            if response.success:
                self.get_logger().info(f"Servo on channel {channel} set to PWM {pwm_value}")
            else:
                self.get_logger().info("Failed to set servo.")
        except Exception as e:
            self.get_logger().info("Service call failed: {e}")


    # GOTTA FIX EVERYTHING UNDER HERE
    ####################################

    def request_drone_data(self):
        rospy.wait_for_service("/get_drone_data")
        try:
            get_data = rospy.ServiceProxy("/get_drone_data", GetGPSData)
            response = get_data()
            return GetGPSDataResponse(
                response.latitude, response.longitude, response.altitude, response.yaw
            )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def delete_waypoint_data(self, index):
        rospy.loginfo("called waypoint deletion function")
        rospy.wait_for_service("/DelWaypoint")
        rospy.loginfo("deletion service loaded")
        try:
            del_data = rospy.ServiceProxy("/DelWaypoint", DelWaypoint)
            request = DelWaypointRequest()
            request.index = index
            response = del_data(request)
            return DelWaypointResponse(response.success)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def send_waypoint_data(self, lat, long, alt, index):
        rospy.loginfo("called waypoint function")
        rospy.wait_for_service("/AddWaypoint")
        rospy.loginfo("addition service loaded")
        try:
            send_data = rospy.ServiceProxy("/AddWaypoint", AddWaypoint)
            request = AddWaypointRequest()
            request.altitude = alt
            request.longitude = long
            request.latitude = lat
            request.index = index
            response = send_data(request)
            return AddWaypointResponse(response.success)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def activate_servo(self):
        servo_channel = 9
        self.set_servo(servo_channel, 1500)
        time.sleep(1)
        self.set_servo(servo_channel, 1000)

    def change_mode(self, mode):
        rospy.loginfo(f"Setting mode to {mode}...")
        response = self.set_mode(custom_mode=mode)

        if response.mode_sent:
            rospy.loginfo(f"Mode changed to {mode}")
        else:
            rospy.logerr("Failed to change mode")

    def update_waypoint_reached(self, msg):
        self.waypoint_reached = msg.wp_seq

        if self.waypoint_reached == 2:
            self.lap += 1
            q = self.detected_object_waypoints.get_detected_objects()
            self.get_logger().info(f"{BLUE}Queue Size: {len(q)}, First Object: {q[0]['name'] if q else 'None'}{RESET}")
            
            if self.lap >= 2:
                q = self.detected_object_waypoints.get_detected_objects()

                if self.lap > 2:
                    name = q[0]["name"]
                    index = q[0]["index"]
                    message = f"{name} waypoint DELETED at index {index}"
                    self.send_status(message)
                    self.delete_waypoint_data(index)
                    self.detected_object_waypoints.rotate_waypoints()
                    q = self.detected_object_waypoints.get_detected_objects()

                    if (
                        self.lap >= 6
                    ):  # can change this based on how many laps we wanna do, so if we wanna do 5 laps, this would be +1
                        rospy.loginfo("Mission over, returning back to home")
                        message = "Mission over, returning back to home"
                        self.send_status(message)
                        self.change_mode("RTL")
                        return

                name = q[0]["name"]
                lat = q[0]["latitude"]
                long = q[0]["longitude"]
                index = q[0]["index"]
                message = f"{name} INSERTED at index: {index}"
                self.send_status(message)
                self.change_mode("GUIDED")
                self.send_waypoint_data(lat, long, ALT, index)
                self.change_mode("AUTO")

            rospy.loginfo(f"{GREEN}Lap Updated: {self.lap}{RESET}")