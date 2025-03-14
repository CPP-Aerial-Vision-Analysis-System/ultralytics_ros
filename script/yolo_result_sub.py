#!/usr/bin/env python3
import rospy
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2D
from mavros_msgs.srv import CommandLong, CommandLongRequest, CommandLongResponse, SetMode
from geometry_msgs.msg import Pose2D
import time, cv2, math, sys
from gps_mavros.srv import GetGPSData, GetGPSDataResponse
from waypoint_mavros.srv import AddWaypointResponse, AddWaypoint, AddWaypointRequest
#from  camera_frame import WaypointManager

class YoloResultSubscriber:

    def __init__(self):
        self.subscriber = rospy.Subscriber(
            "yolo_result", YoloResult, self.callback, queue_size=1
        )

        # rospy.wait_for_service("/mavros/cmd/command")
        # self.command_service = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)
        #self.waypoint_manager= WaypointManager()

        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
       
        self.lasttime = time.time()
        self.run_detection_once = False
        self.waypoint_reached = 0
    
    def update_waypoint_reached(self,msg):
        self.waypoint_reached = msg

    def gps_calc(self, gps_lat, gps_lon, target_x, target_y, img_width, img_height, fov_width, fov_height, altitude, yaw_degrees):
        """
        Calculate the GPS coordinates of a target point in an image taken from a drone, accounting for:
        - Drone's GPS position (gps_lat, gps_lon)
        - Target's pixel position in the image (target_x, target_y)
        - Camera specifications (img_width, img_height, fov_width, fov_height)
        - Drone's altitude
        - Drone's yaw orientation (yaw_degrees, 0° = North)

        Returns:
            new_gps_lat, new_gps_lon
        """
        # Convert FOV from degrees to radians
        fov_width_rad = math.radians(fov_width)
        fov_height_rad = math.radians(fov_height)

        # Compute ground coverage based on altitude
        ground_width = 2 * altitude * math.tan(fov_width_rad / 2)
        ground_height = 2 * altitude * math.tan(fov_height_rad / 2)

        # Calculate meters per pixel
        meters_per_pixel_x = ground_width / img_width
        meters_per_pixel_y = ground_height / img_height

        # Compute pixel displacement from the image center
        image_center_x = img_width / 2.0
        image_center_y = img_height / 2.0

        dx_pixels = target_x - image_center_x
        dy_pixels = image_center_y - target_y  # Invert because image Y axis is top-down

        # Convert pixel displacement to real-world displacement (in meters)
        dx_meters = dx_pixels * meters_per_pixel_x
        dy_meters = dy_pixels * meters_per_pixel_y

        # Convert yaw from degrees to radians for rotation
        yaw_radians = math.radians(yaw_degrees)

        # Rotate displacement based on drone yaw (2D rotation matrix)
        rotated_dx = dx_meters * math.cos(yaw_radians) - dy_meters * math.sin(yaw_radians)
        rotated_dy = dx_meters * math.sin(yaw_radians) + dy_meters * math.cos(yaw_radians)

        # Convert meters to GPS degrees
        meters_per_degree_lat = 111320  # Approximate meters per degree latitude
        meters_per_degree_lon = 111320 * math.cos(math.radians(gps_lat))  # Adjust for longitude scaling

        # Apply rotated displacement to GPS coordinates
        new_gps_lat = gps_lat + (rotated_dy / meters_per_degree_lat)
        new_gps_lon = gps_lon + (rotated_dx / meters_per_degree_lon)

        return new_gps_lat, new_gps_lon

    def callback(self, msg):
        if msg.detections.detections:
            rospy.loginfo(f"{len(msg.detections.detections)} object(s) detected!")
            gps_response = self.request_drone_data()
            
            #print(type(gps_response))
            bbox_coords = msg.detections.detections
            print(gps_response.latitude, gps_response.longitude, gps_response.altitude)
            print(f"Reached {self.waypoint_reached}")

            if self.run_detection_once == False: #time.time() - self.lasttime > 10 :
                #self.lasttime = time.time()
                self.run_detection_once = True
                for i in range(len(bbox_coords)):
                    #rospy.loginfo(bbox_coords[i].bbox.center)
                    #print(gps_response.latitude, gps_response.longitude, gps_response.altitude)
                    lat, long = self.gps_calc(gps_response.latitude,
                                                gps_response.longitude,
                                                bbox_coords[i].bbox.center.x,
                                                bbox_coords[i].bbox.center.y,
                                                640,
                                                640,
                                                55,
                                                55,
                                                gps_response.altitude,
                                                gps_response.yaw)
                    rospy.loginfo("calling waypoint service")
                    waypoint_response = self.send_waypoint_data(lat, long, gps_response.altitude)
                    #rospy.loginfo(f"Waypoint: {waypoint_response.success}")
                    if(waypoint_response.success == True):
                        rospy.loginfo(f"Calculated Position: LAT: {lat}, LONG:{long}")
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
        rospy.wait_for_service('/get_drone_data')
        try:
            get_data = rospy.ServiceProxy('/get_drone_data', GetGPSData)
            response = get_data()
            return GetGPSDataResponse(response.latitude, response.longitude, response.altitude, response.yaw)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def send_waypoint_data(self,lat,long,alt):
        rospy.loginfo("called waypoint function")
        rospy.wait_for_service('/AddWaypoint')
        rospy.loginfo("service loaded")
        try:
            send_data = rospy.ServiceProxy('/AddWaypoint', AddWaypoint)
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

    def set_mode(self):
        rospy.loginfo("Setting mode to GUIDED...")
        response = self.set_mode(custom_mode = "GUIDED")

        if response.mode_sent:
            rospy.loginfo("Mode changed to GUIDED")
        else:
            rospy.logerr("Failed to change mode")


if __name__ == "__main__":
    rospy.init_node("yolo_result_subscriber")
    node = YoloResultSubscriber()
    node.set_mode()
    rospy.spin()
