#!/usr/bin/env python3
import rospy
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2D
from mavros_msgs.srv import CommandLong, CommandLongRequest, CommandLongResponse
import time, cv2
from gps_mavros.srv import GetGPSData


class YoloResultSubscriber:
    def __init__(self):
        self.subscriber = rospy.Subscriber(
            "yolo_result", YoloResult, self.callback, queue_size=1
        )

        # rospy.wait_for_service("/mavros/cmd/command")
        # self.command_service = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)

    def callback(self, msg):
        if msg.detections.detections:
            rospy.loginfo(f"{len(msg.detections.detections)} object(s) detected!")
            self.request_drone_data()
            # self.activate_servo()
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
            rospy.loginfo(
                f"GPS: ({response.latitude}, {response.longitude}, {response.altitude}), Yaw: {response.yaw}°"
            )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def activate_servo(self):
        servo_channel = 9
        self.set_servo(servo_channel, 1500)
        time.sleep(1)
        self.set_servo(servo_channel, 1000)


if __name__ == "__main__":
    rospy.init_node("yolo_result_subscriber")
    node = YoloResultSubscriber()
    rospy.spin()
