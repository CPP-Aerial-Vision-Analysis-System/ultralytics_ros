#!/usr/bin/env python3

import rospy
import time
from pymavlink import mavutil
from mavros_msgs.msg import StatusText
from mavros_msgs.srv import SetMode

class ServoController:
    # === CONFIGURATION ===
    SERVO_PULLEY = 9       # AUX1
    SERVO_SCISSOR = 10     # AUX2
    PULLEY_OPEN = 900
    PULLEY_CLOSE = 750
    SCISSOR_CUT = 1800
    BUTTON_BIT = 2         # Bit 2 = AUX3 = SERVO11

    def __init__(self):
        print("Initializing connection...")
        self.master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=115200)
        self.master.wait_heartbeat()
        print("Connected to Pixhawk.")

        rospy.loginfo("Starting ServoController node")

        self.status_pub = rospy.Publisher("/mavros/statustext/send", StatusText, queue_size=10)
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    def move_servo(self, channel, pwm):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, channel, pwm, 0, 0, 0, 0, 0
        )
        print(f"[SERVO] Channel {channel} moved to {pwm}μs")

    def run_pulley_sequence(self, cycles=3):
        for i in range(cycles):
            print(f"Cycle {i+1}/{cycles}: Opening pulley")
            self.move_servo(self.SERVO_PULLEY, self.PULLEY_OPEN)
            time.sleep(1.25)
            print(f"Cycle {i+1}/{cycles}: Closing pulley")
            self.move_servo(self.SERVO_PULLEY, self.PULLEY_CLOSE)
            time.sleep(0.5)
        print("Pulley sequence done.")
        message = f"Pulley sequence done"
        self.send_status(message)

    def cut_with_scissors(self):
        print("Waiting 10 seconds before cutting with scissors...")
        time.sleep(10)
        self.move_servo(self.SERVO_SCISSOR, self.SCISSOR_CUT)
        print("Scissors cut.")
        message = f"Scissors cut"
        self.send_status(message)

    def wait_for_limit_switch_release(self):
        print("Monitoring limit switch release (AUX3/SERVO11)...")
        last_state = None
        while True:
            msg = self.master.recv_match(type='BUTTON_CHANGE', blocking=True, timeout=10)
            if msg:
                state = (msg.state >> self.BUTTON_BIT) & 1
                if last_state is None:
                    last_state = state
                if last_state == 0 and state == 1:
                    print("Limit switch released: sequence complete.")
                    message = f"Limit switched released: sequence complete"
                    self.send_status(message)
                    time.sleep(10)
                    self.change_mode("AUTO")  # Change to AUTO mode after release
                    return
                last_state = state
            else:
                print("No BUTTON_CHANGE received in 10 seconds")
                message = f"No BUTTON_CHANGE received in 10 seconds"
                self.send_status(message)

    def execute(self):
        self.run_pulley_sequence()
        self.cut_with_scissors()
        self.wait_for_limit_switch_release()
    
    def send_status(self, text, throttle=False):
        now = time.time()

        if not throttle or (now - self.last_status_time > self.status_interval):
            status_msg = StatusText()
            status_msg.severity = 6  # 6 = NOTICE
            status_msg.text = text
            self.status_pub.publish(status_msg)
            self.last_status_time = now
    
    def change_mode(self, mode):
        rospy.loginfo(f"Setting mode to {mode}...")
        response = self.set_mode(custom_mode=mode)

        if response.mode_sent:
            rospy.loginfo(f"Mode changed to {mode}")
        else:
            rospy.logerr("Failed to change mode")


if __name__ == "__main__":
    try:
        controller = ServoController()
        controller.execute()
    except KeyboardInterrupt:
        print("\nExiting.")
