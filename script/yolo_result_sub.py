#!/usr/bin/env pythom3
import rclpy
from rclpy.node import Node
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2D
from mavros_msgs.srv import CommandLong, SetMode, WaypointSetCurrent, WaypointPull
from mavros_msgs.msg import WaypointReached, VfrHud, StatusText
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import NavSatFix, Image
from std_msgs.msg import Bool
from rcl_interfaces.srv import GetParameters

# from cv_bridge import CvBridge        | check if this is in ros2

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

# both in degrees
HFOV = 68.75
VFOV = 53.13

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

    def clear_queue(self):
        return self.detected_objects.clear()

class YoloResultSubscriber(Node):

    def __init__(self):
        super().__init__('yolo_result_subscriber')

        # Subscribers
        self.create_subscription(YoloResult, 'yolo_result', self.yolo_result_cb, 1)
        self.create_subscription(WaypointReached, "/mavros/mission/reached", self.update_waypoint_reached, 1)   # original doesn't have queue
        self.create_subscription(VfrHud, "/mavros/vfr_hub", self.speed_cb)
        self.create_subscription(StatusText, "/mavros/statustext/recv", self.restart_callback)

        # Publishers
        self.status_pub = self.create_publisher(StatusText, "/mavros/statustext/send", queue=10)
        self.detected_photo_pub = self.create_publisher(Bool, "/camera/object_detected", queue=10)

        # Clients
        self.set_wp_client = self.create_client(WaypointSetCurrent, "/mavros/mission/set_current")
        self.waypoint_pull_client = self.create_client(WaypointPull, "/mavross/mission/pull")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        
        self._wait_for_services()   # wait for services to be available

        # Wait for all services to be available
        def _wait_for_services(self):
            clients = [
                ('/mavros/mission/set_current', self.set_wp_client),
                ('/mavross/mission/pull', self.waypoint_pull_client),
                ('/mavros/set_mode', self.set_mode_client)
            ]
            for name, client in clients:
                while not client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info(f'{name} service not available, waiting...')

        # variables
        self.detected_object_waypoints = Detected_Object_Waypoints()
        self.last_before_rtl = 0
        self.next_after_takeoff = 0
        self.takeoff_index = 0
        self.rtl_index = 0
        self.lap = 0

        # self.servo_controller = ServoController()

        self.last_status_time = 0
        self.status_interval = 8        # seconds between GCS messages

        self.lasttime = time.time()
        self.run_detection_once = False
        self.waypoint_reached = 0

        self.param_manager = ParameterManager()
        class_names = self.param_manager.get_param(self.param_manager.tracknode_client, list_params=['yolo_class_names'])['yolo_class_names']       # not tested

        while class_names is None:
            self.get_logger().warn(f"Waiting for /yolo_class_names to be set...")
            class_names = self.param_manager.get_param(self.param_manager.tracknode_client, list_params=['yolo_class_names'])['yolo_class_names']
        self.class_names = eval(class_names)

        self.fetch_mission_indices()

        self.latest_yolo_image_msg = None
        # self.bridge = CvBridge()        
        # self.bridgeObject = CvBridge()
        rp = rospkg.RosPack()
        self.create_subscription(Image, "/yolo_image", self.yolo_image_callback)
        self.detected_object_path = os.path.join(rp.get_path("video_cam"), "detected")

        if not os.path.exists(self.detected_object_path):
            os.makedirs(self.detected_object_path)

        self.latest_image_msg = None        # check are we using Yolo still if not then might need new Image msg?

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

        # self.GEOFENCE2 = {
        #     "min_lat": min_lat,
        #     "max_lat": max_lat,
        #     "min_lon": min_lon,
        #     "max_lon": max_lon,
        #     # "alt": alt  
        # }

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
            message = f"within geofence"
            self.send_status(message, True)
        else:
            print(f"{YELLOW}Geofence status: Outside{RESET}")   # fix
            message = f"NOT within geofence"
            self.send_status(message, True)

    def fetch_mission_indices(self):
        wp_params = ['num_waypoints', 'takeoff_index', 'rtl_index', 'next_after_takeoff', 'last_before_rtl']
        params = self.param_manager.get_param(self.param_manager.waypoint_client, list_params=wp_params)
        num_waypoints, takeoff_index, rtl_index, next_after_takeoff, last_before_rtl = [int(params[k] for k in wp_params)]
        
        self.num_waypoints = int(num_waypoints)
        self.takeoff_index = int(takeoff_index)
        self.rtl_index = int(rtl_index)
        self.next_after_takeoff = int(next_after_takeoff)
        self.last_before_rtl = int(last_before_rtl)

    def sim_image_callback(self, msg):
        self.latest_image_msg = msg
    
    def yolo_image_callback(self, msg):
        self.latest_yolo_image_msg = msg
    
    def pull_waypoints(self):
        """Update the mission on the fcu"""
        while not self.waypoint_pull_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for waypoint pull service...")
        
        try:
            future = self.waypoint_pull_client.call_async(WaypointPull.Request())
            rclpy.spin_until_future_complete(self, future)
            self.get_logger().info(f"Waypoint pull success: {future.result().wp_received}")     # maybe change the response
        except Exception as e:
            self.get_logger().warn("Failed to pull waypoints")
            self.get_logger().error(str(e))

    def restart_callback(self, msg):
        if "restart" in msg.text.lower():
            message = f"Restarting code"
            self.send_status(message, False)
            self.waypoint_reached = 0
            self.lap = 0
            self.run_detection_once = False
            self.detected_object_waypoints.clear_queue()
            self.get_logger().info(f"Waypoint reached = {self.waypoint_reached}")
            self.get_logger().info(f"Lap = {self.lap}")
            self.get_logger().inf(f"Run Detection Once = {self.run_detection_once}")
            q = self.detected_object_waypoints.get_detected_objects()
            self.get_logger().info(f"Queue Size = {len(q)}, Objects in Queue: {q}")
            self.update_mission_data()
            self.fetch_mission_indices()

    def trigger_camera(self):
        self.get_logger().info("Object Detected. Triggering Jetson-side camera")
        self.detected_photo_pub.publish(Bool(data=True))

    def update_waypoint_reached(self, msg):
        self.waypoint_reached = msg.wp_seq
        q = self.detected_object_waypoints.get_detected_objects()

        if self.waypoint_reached = self.last_before_rtl:
            self.lap += 1

            if len(q) = 0:
                self.get_logger().info("Queue Empty")
                index = self.rtl_index
                self.delete_waypoint_data(index)
                return
            
            name = q[0]['name']
            lat = q[0]["latitude"]
            long = q[0]["longitude"]
            index = self.last_before_rtl + 1
            message = f"{name} INSERTED at index: {index}"
            self.send_status(message, False)
            self.change_mode("GUIDED")
            self.send_waypoint_data(lat, long, ALT, index)
            self.change_mode("AUTO")

            self.rtl_index = index
            self.last_before_rtl = -1
            return
        
        if self.waypoint_reached == self.rtl_index:
            self.get_logger().info(f"{GREEN}Object waypoint reached{RESET}")
            message = f"Object waypoint reached. Payload dropping"
            self.send_status(message, False)
            self.change_mode("GUIDED")
            self.delete_waypoint_data(self.rtl_index + 1)
            self.change_mode("AUTO")
            self.change_mode("GUIDED")
            # self.servo_controller.run_sequence()
            rospy.loginfo("Stopping detection...")
            self.subscriber.unregister()
        
        self.get_logger().info(f"{GREEN}Lap Updated: {self.lap}{RESET}")

    def speed_cb(self, msg):
        self.get_logger().info(f"{BLUE}Current airspeed: {msg.airspeed:.2f}{RESET}")

    def gps_calc(self, gps_lat, gps_lon, target_x, target_y, img_width, img_height, yaw_degrees):
        """Calculate GPS coordinates assuming max shift of 100ft (~0.00030 deg) from image center to edge."""
        # Max GPS shift from center to edge (in degrees)
        max_deg_shift = 0.00001373  # ~5 feet

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

    def yolo_result_cb(self, msg):
        if msg.detections.detections:
            self.get_logger().info(f"{len(msg.detections.detections)} object(s) detected!")
            gps_response = self.request_drone_data()

            # print(type(gps_response))
            bbox_coords = msg.detections.detections
            self.get_logger().debug(gps_response.latitude, gps_response.longitude, gps_response.altitude)
            self.get_logger().info_throttle(5.0, f"Current wp_reached {self.waypoint_reached}")

            if self.within_geofence:        # time.time() - self.lasttime > 10:
                # self.lasttime = time.time()
                # self.run_detection_once = True
                for i in range(len(bbox_coords)):
                    # self.get_logger().info(bbox_coords[i].bbox.center)
                    # print(gps_response.latitude, gps_response.longitude, gps_response.altitude)
                    lat, long = self.gps_calc(
                        gps_response.latitude,
                        gps_response.longitude,
                        bbox_coords[i].bbox.center.x,
                        bbox_coords[i].bbox.center.y,
                        640,
                        480,
                        gps_response.yaw
                    )
                    # self.get_logger().info("calling waypoint service")
                    # waypoint_response = self.send_waypoint_data(
                    #     lat, long, 50
                    # )
                    # self.get_logger.info(f"Waypoint: {waypoint_response.success}")
                    detected_name = self.class_names[bbox_coords[i].results[0].id]
                    index = max(self.next_after_takeoff, self.waypoint_reached + 1)
                    if not self.compare_object_names(detected_name):
                        message = f"{detected_name} DETECTED at index {index}"
                        self.send_status(message, False)
                        self.get_logger().info(f"Calculated Position: LAT: {lat}, LONG: {long}")
                        self.detected_object_waypoints.add_object(
                            self.class_names[bbox_coords[i].results[0].id],
                            lat,
                            long,
                            ALT,
                            index
                        )
                        # self.trigger_camera()
                        queue_length = len(
                            self.detected_object_waypoints.get_detected_objects()
                        )
                        if self.latest_yolo_image_msg is not None and queue_length <= 2:
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            # yolo_image = self.bridge.imgmsg_to_cv2(
                            #     self.latest_yolo_image_msg, desired_encoding="bgr8"
                            # )
                            detected_filename = os.path.join(
                                self.detected_object_path,
                                f"detected_photo_{timestamp}.jpg"
                            )
                            cv2.imwrite(detected_filename, yolo_image)
                            self.get_logger().info(f"Photo saved to {detected_filename}")

                        # message = (
                        #     f"'{detected_name}' at " f"LAT: {lat:.6f}, LON: {long:.6f}"
                        # )
                        # self.send_status(message)
                        # message = f"WP added at {index}"
                        # self.send_status(message)
                        # self.change_mode("GUIDED")
                        # self.send_waypoint_data(lat, long, ALT, index)
                        # self.change_mode("AUTO")
                        self.get_logger().info(self.detected_object_waypoints.get_detected_objects())
            else:
                self.get_logger().info_throttle(5, "No objects detected.")

    def compare_object_names(self, object_name):
        q = self.detected_object_waypoints.get_detected_objects()
        return any(item["name"] == object_name for item in q)

    # def set_servo(self, channel, pwm_value):
    #     try:
    #         command = CommandLong.Request()
    #         command.command = 183  # MAV_CMD_DO_SET_SERVO
    #         command.param1 = channel
    #         command.param2 = pwm_value
    #         response = self.command_service(command)            # maybe error here
    #         if response.success:
    #             self.get_logger().info(f"Servo on channel {channel} set to PWM {pwm_value}")
    #         else:
    #             self.get_logger().info("Failed to set servo.")
    #     except Exception as e:
    #         self.get_logger().info("Service call failed: {e}")

    def request_drone_data(self):
        self.drone_client = self.create_client(GetGPSData, "/get_drone_data")
        while not self.drone_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for GPS data...")
        
        try:
            future = self.drone_client.call_async(GetGPSData.Request())
            rclpy.spin_until_future_complete(self, future)
            return future.result()
        except Exception as e:
            self.get_logger().error(str(e))

    def update_mission_data(self):
        self.get_logger().info("called mission update function")
        # self.update_mission_client = self.create_client(UpdateMission, "/UpdateMission")
        while not self.update_mission_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for Update Mission service...")
        
        try:
            future = self.update_mission_client.call_async(UpdateMission.Request())
            rclpy.spin_until_future_complete(self, future)
            return future.result()
        except Exception as e:
            self.get_logger().error(str(e))

    def delete_waypoint_data(self, index):
        self.get_logger().info("called waypoint delete function")
        self.delete_wp_client = self.create_client(DelWayPoint, "/DelWaypoint")
        while not self.delete_wp_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for delete wp service...")
        self.get_logger().info("deletion service loaded")
        try:
            req = DelWayPoint.Request()
            req.index = index
            future = self.delete_wp_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            return future.result()
        except Exception as e:
            self.get_logger().error(str(e))

    def send_waypoint_data(self, lat, long, alt, index):
        self.get_logger().info("called waypoint function")
        self.add_wp_client = self.create_client(AddWaypoint, "/AddWaypoint")
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

    # def activate_servo(self):
    #     servo_channel = 9
    #     self.set_servo(servo_channel, 1500)
    #     time.sleep(1)
    #     self.set_servo(servo_channel, 1000)

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

    # TODO: fix throttle. function still works but throttling needs to be fixed
    def send_status(self, text, throttle=False):
        now = time.time()

        if throttle and (now - self.last_status_time <= self.status_interval):
            return
        status_msg = StatusText()
        status_msg.severity = 6     # 6 = NOTICE
        status_msg.text = text
        self.status_pub.publish(status_msg)
        self.last_status_time = now

    def set_mission_index(self, index):
        # set_wp_client service should already be ready from self._wait_for_services
        self.get_logger().info(f"Setting current mission waypoint to {index}")
        try:
            req = WaypointSetCurrent.Request()
            req.wp_seq = index
            future = self.set_wp_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if response.success:
                self.get_logger().info(f"Mission set to waypoint {index}")
            else:
                raise Exception("Failed to set mission index.")
        except Exception as e:
            self.get_logger().error(str(e))
