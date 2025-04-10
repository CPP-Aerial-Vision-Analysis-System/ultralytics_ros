#!/usr/bin/env python3
import rospy
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2D
from mavros_msgs.srv import (
    CommandLong,
    CommandLongRequest,
    CommandLongResponse,
    SetMode,
)
from mavros_msgs.msg import WaypointReached
from geometry_msgs.msg import Pose2D
import time, cv2, math, sys
from gps_mavros.srv import GetGPSData, GetGPSDataResponse
from waypoint_mavros.srv import AddWaypointResponse, AddWaypoint, AddWaypointRequest
from collections import deque

# from  camera_frame import WaypointManager


class Detected_Object_Waypoints:
    def __init__(self):
        self.detected_objects = (
            deque()
        )  # Queue to store detected objects and their GPS waypoints

    def add_object(self, object_name, lat, long, alt):
        """
        Adds a detected object with its name and calculated GPS coordinates to the queue.
        """
        detected_object = {
            "name": object_name,
            "latitude": lat,
            "longitude": long,
            "alt": alt,
        }
        self.detected_objects.append(detected_object)
        rospy.loginfo(
            f"Object '{object_name}' added at LAT: {lat}, LONG: {long}, ALT: {alt}"
        )

    def get_detected_objects(self):
        """
        Returns the queue of detected objects with their GPS waypoints.
        """
        return self.detected_objects


class YoloResultSubscriber:

    def __init__(self):
        self.subscriber = rospy.Subscriber(
            "yolo_result", YoloResult, self.callback, queue_size=1
        )
        rospy.Subscriber(
            "/mavros/mission/reached", WaypointReached, self.update_waypoint_reached
        )

        # rospy.wait_for_service("/mavros/cmd/command")
        # self.command_service = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)
        # self.waypoint_manager= WaypointManager()

        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.lasttime = time.time()
        self.run_detection_once = False
        self.waypoint_reached = 0
        # self.object_waypoints = deque()
        self.detected_object_waypoints = Detected_Object_Waypoints()

        class_names = rospy.get_param("/yolo_class_names", None)
        while class_names is None:
            rospy.logwarn("Waiting for /yolo_class_names to be set...")
            rospy.sleep(0.5)
            class_names = rospy.get_param("/yolo_class_names", None)
        self.class_names = eval(class_names)

    def update_waypoint_reached(self, msg):
        self.waypoint_reached = msg.wp_seq
        # if self.detected_object_waypoints and (msg.latitude, msg.longitude) == self.get_detected_objects:
        #     rospy.loginfo("Object waypoint reached. Switching to LOITER mode.")
        #     self.set_mode("LOITER")
        #     rospy.sleep(15)  # Hold at waypoint for 15 seconds
        #     rospy.loginfo("Resuming flight path. Switching back to AUTO mode.")
        #     self.set_mode("AUTO")

    def gps_calc(
        self, gps_lat, gps_lon, target_x, target_y, img_width, img_height, yaw_degrees
    ):
        """
        Calculate GPS coordinates assuming max shift of 100ft (~0.00030 deg) from image center to edge.
        """

        # Max GPS shift from center to edge (in degrees)
        max_deg_shift = 0.00030

        # Compute center of the image
        image_center_x = img_width / 2.0
        image_center_y = img_height / 2.0

        # Pixel displacement from center
        dx_pixels = target_x - image_center_x
        dy_pixels = (
            target_y - image_center_y
        )  # don't invert; use image convention consistently

        # Max possible pixel distance (diagonal from center to corner)
        max_pixel_distance = math.sqrt((image_center_x) ** 2 + (image_center_y) ** 2)

        # Actual pixel distance from center to target
        actual_pixel_distance = math.sqrt(dx_pixels**2 + dy_pixels**2)

        # Normalize displacement (0 to 1 scale)
        norm_dx = dx_pixels / max_pixel_distance
        norm_dy = dy_pixels / max_pixel_distance

        # Scale normalized values to GPS degree shift (max 0.00030 degrees)
        raw_shift_lon = norm_dx * max_deg_shift
        raw_shift_lat = norm_dy * max_deg_shift

        # Apply yaw rotation (so direction matches drone orientation)
        yaw_rad = math.radians(yaw_degrees)
        rotated_lon = raw_shift_lon * math.cos(yaw_rad) - raw_shift_lat * math.sin(
            yaw_rad
        )
        rotated_lat = raw_shift_lon * math.sin(yaw_rad) + raw_shift_lat * math.cos(
            yaw_rad
        )

        # Apply shift to original GPS coordinates
        new_gps_lat = gps_lat - rotated_lat
        new_gps_lon = gps_lon + rotated_lon

        return new_gps_lat, new_gps_lon

    def callback(self, msg):
        if msg.detections.detections:
            rospy.loginfo(f"{len(msg.detections.detections)} object(s) detected!")
            gps_response = self.request_drone_data()

            # print(type(gps_response))
            bbox_coords = msg.detections.detections
            print(gps_response.latitude, gps_response.longitude, gps_response.altitude)
            print(f"Current wp_reached {self.waypoint_reached}")

            if self.run_detection_once == False:  # time.time() - self.lasttime > 10 :
                # self.lasttime = time.time()
                self.run_detection_once = True
                for i in range(len(bbox_coords)):
                    # rospy.loginfo(bbox_coords[i].bbox.center)
                    # print(gps_response.latitude, gps_response.longitude, gps_response.altitude)
                    lat, long = self.gps_calc(
                        gps_response.latitude,
                        gps_response.longitude,
                        bbox_coords[i].bbox.center.x,
                        bbox_coords[i].bbox.center.y,
                        640,
                        640,
                        gps_response.yaw,
                    )
                    rospy.loginfo("calling waypoint service")
                    waypoint_response = self.send_waypoint_data(
                        lat, long, gps_response.altitude
                    )
                    # rospy.loginfo(f"Waypoint: {waypoint_response.success}")
                    if waypoint_response.success == True:
                        rospy.loginfo(f"Calculated Position: LAT: {lat}, LONG:{long}")
                        self.detected_object_waypoints.add_object(
                            self.class_names[bbox_coords[i].results[0].id],
                            lat,
                            long,
                            50,
                        )
                        print(self.detected_object_waypoints.get_detected_objects())
                        # find out where the name of the object is stored
                    else:
                        rospy.logwarn("Waypoint not calculated")
        else:
            rospy.loginfo("No objects detected.")

    def set_servo(self, channel, pwm_value):
        try:
            command = CommandLongRequest()
            command.command = 183  # MAV_CMD_DO_SET_SERVO
            command.param1 = channel
            command.param2 = pwm_value
            response = self.command_service(command)
            if response.success:
                rospy.loginfo(f"Servo on channel {channel} set to PWM {pwm_value}")
            else:
                rospy.logerr("Failed to set servo.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

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

    def send_waypoint_data(self, lat, long, alt):
        rospy.loginfo("called waypoint function")
        rospy.wait_for_service("/AddWaypoint")
        rospy.loginfo("service loaded")
        try:
            send_data = rospy.ServiceProxy("/AddWaypoint", AddWaypoint)
            request = AddWaypointRequest()
            request.altitude = alt
            request.longitude = long
            request.latitude = lat
            response = send_data(request)
            return AddWaypointResponse(response.success)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def activate_servo(self):
        servo_channel = 9
        self.set_servo(servo_channel, 1500)
        time.sleep(1)
        self.set_servo(servo_channel, 1000)

    def set_mode(self, mode):
        rospy.loginfo(f"Setting mode to {mode}...")
        response = self.set_mode(custom_mode=mode)

        if response.mode_sent:
            rospy.loginfo(f"Mode changed to {mode}")
        else:
            rospy.logerr("Failed to change mode")


if __name__ == "__main__":
    rospy.init_node("yolo_result_subscriber")
    node = YoloResultSubscriber()
    rospy.spin()
