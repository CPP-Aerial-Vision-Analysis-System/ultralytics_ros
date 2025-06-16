#!/usr/bin/env python3

import rospy
import time
import math
import Jetson.GPIO as GPIO
from mavros_msgs.srv import CommandLong, SetMode
from mavros_msgs.msg import RCIn, StatusText
from std_msgs.msg import Float64

SERVO_PULLEY = 9       # AUX1 = Servo 9
SERVO_SCISSOR = 10     # AUX2 = Servo 10
PULLEY_OPEN = 925       #1050
PULLEY_CLOSE = 725      #850
SCISSOR_OPEN = 1800
SCISSOR_CUT = 800

#HARD-CODE PARAMS
NUM_CYCLES = 5
OPEN_TIME = 1.1  # Time to open pulley in seconds
CLOSE_TIME = 2  # Time to close pulley in seconds
PULLEY_RADIUS = 1.3 # inches   #1.22
PULLEY_RADIUS_FT = PULLEY_RADIUS / 12.0
CIRCUMFERENCE_FT = 2*math.pi * PULLEY_RADIUS_FT
DROP_INTERRUPT_FT = 47 #45
TICKS_MAX = 36  # around 18 full rotations
# GPIO pin configuration
LIMIT_SWITCH_PIN = 29          # Physical pin on Jetson board (BOARD mode)
HALL_SENSOR_PIN = 15

class ServoController:
    def __init__(self):
        #rospy.init_node('mavros_servo_control', anonymous=True)
        rospy.wait_for_service('/mavros/cmd/command')
        self.command_srv = rospy.ServiceProxy('/mavros/cmd/command', CommandLong)
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.status_pub = rospy.Publisher("/mavros/statustext/send", StatusText, queue_size=10)
        
        self.last_status_time = 0
        self.status_interval = 5  # Throttle interval in seconds

        # GPIO setup
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(LIMIT_SWITCH_PIN, GPIO.IN)
        GPIO.setup(HALL_SENSOR_PIN, GPIO.IN)
        rospy.on_shutdown(self.cleanup_gpio)

        self.last_hall_state = GPIO.input(HALL_SENSOR_PIN)
        self.tick_count = 0
        self.drop_distance_ft = 0.0
        self.alt = 0

        rospy.Subscriber("/mavros/global_position/rel_alt", Float64, self.altitude_callback)
        rospy.loginfo(f"✅ GPIO initialized. Monitoring pin {LIMIT_SWITCH_PIN} for limit switch.")

    def cleanup_gpio(self):
        GPIO.cleanup()
        rospy.loginfo("🧹 GPIO cleaned up.")

    def move_servo(self, channel, pwm):
        try:
            res = self.command_srv(
                broadcast=False,
                command=183,  # MAV_CMD_DO_SET_SERVO
                confirmation=0,
                param1=channel,
                param2=pwm,
                param3=0, param4=0,
                param5=0, param6=0, param7=0
            )
            if res.success:
                rospy.loginfo(f"[SERVO] Channel {channel} moved to {pwm}μs")
                self.send_status(f"Servo {channel} -> {pwm}")
            else:
                rospy.logwarn(f"[SERVO] Failed to move channel {channel}")
                self.send_status(f"Servo {channel} move FAILED")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def wait_for_limit_switch(self):
        rospy.loginfo("Waiting for limit switch release...")
        self.send_status("Waiting for limit switch release...")
        if GPIO.input(LIMIT_SWITCH_PIN) == GPIO.HIGH:
            rospy.loginfo(f"{GPIO.input(LIMIT_SWITCH_PIN) == GPIO.HIGH}")
            rospy.loginfo("Limit switch STILL PRESSED. Proceeding...")
            self.send_status("Limit switch STILL PRESSED.")
            return True
        else:
            rospy.loginfo("Limit switch RELEASED. Proceeding...")
            self.send_status("Limit switch RELEASED.")
            return False


    def change_mode(self, mode):
        rospy.loginfo(f"Setting mode to {mode}...")
        response = self.set_mode(custom_mode=mode)
        if response.mode_sent:
            rospy.loginfo(f"Mode changed to {mode}")
        else:
            rospy.logerr("Failed to change mode")

    def send_status(self, text, throttle=False):
        now = time.time()
        if not throttle or (now - self.last_status_time > self.status_interval):
            status_msg = StatusText()
            status_msg.severity = 6  # NOTICE
            status_msg.text = text
            self.status_pub.publish(status_msg)
            self.last_status_time = now

    def count_rotations(self):
        """Counts rising edges on hall sensor. Call this repeatedly in your main loop to update count."""
        current_state = GPIO.input(HALL_SENSOR_PIN)
        if self.last_hall_state == GPIO.LOW and current_state == GPIO.HIGH:
            self.tick_count += 1
        self.last_hall_state = current_state

    def altitude_callback(self, msg):
        self.alt = msg.data
        rospy.loginfo(f"Current altitude: {self.alt} m")
        
    def run_sequence(self):
        self.tick_count = 0
        self.drop_distance_ft = 0

        for i in range(NUM_CYCLES):
            ticks_count_cycle_start = self.tick_count
            open = OPEN_TIME
            if(i == NUM_CYCLES - 1):
                try:
                    height = self.alt * 3.28 
                    if height < 25:
                        height = 57

                except:
                    height = 57
                drop_feet = height - DROP_INTERRUPT_FT
                num_revs_needed = drop_feet / CIRCUMFERENCE_FT
                ticks_needed = int(num_revs_needed * 2)  # 2 ticks per revolution
                open = 0.7
                rospy.loginfo(
            f"Last cycle: Altitude={height:.2f} ft, "
            f"Target drop={drop_feet:.2f} ft, "
            f"Num revs needed={num_revs_needed:.2f}, "
            f"Ticks needed={ticks_needed:.2f}, "
            )
            else:
                ticks_needed = TICKS_MAX
            rospy.loginfo(f"Cycle {i+1}/5: Opening pulley")
            self.move_servo(SERVO_PULLEY, PULLEY_OPEN)
            start = time.time()
            while time.time() - start < open:
                self.count_rotations()
                ticks_this_cycle = self.tick_count - ticks_count_cycle_start
                if ticks_this_cycle >= ticks_needed:
                    break
                time.sleep(0.001)  # Adjust sleep time as needed for your application

            self.move_servo(SERVO_PULLEY, PULLEY_CLOSE)
            rospy.loginfo(f"Cycle {i+1}/3: Closing pulley") 
            time.sleep(CLOSE_TIME)

            ticks_this_cycle = self.tick_count - ticks_count_cycle_start
            cycle_revolutions = ticks_this_cycle / 2.0  # 2 ticks per revolution
            cycle_drop = cycle_revolutions * CIRCUMFERENCE_FT
            self.drop_distance_ft += cycle_drop

            status = (
                f"[CYCLE {i+1}] Hall rotations: {cycle_revolutions}, "
                f"[CYCLE {i+1}] Drop Distance:  {cycle_drop:.2f} ft"
                f"Total Drop Distance: {self.drop_distance_ft:.2f} ft"
            )
            print(status)
            rospy.loginfo(status)
            self.send_status(status)

        rospy.loginfo("SEQUENCE DONE. Waiting 5s before scissors cut...")
        self.send_status("SEQUENCE. Waiting 5s for scissors cut...")
        time.sleep(5)

        self.move_servo(SERVO_SCISSOR, SCISSOR_CUT)
        rospy.loginfo("Scissors cut.")
        self.send_status("Scissors cut.")
        time.sleep(1)

        # Wait for limit switch to be released before AUTO mode
        while self.wait_for_limit_switch():
                rospy.loginfo("Limit Switch NOT RELEASED. Initiating Cut Sequence.")
                self.send_status("Limit Switch NOT RELEASED. Initiating Cut Sequence.")
                self.move_servo(SERVO_SCISSOR, SCISSOR_OPEN)
                time.sleep(3)
                self.move_servo(SERVO_SCISSOR, SCISSOR_CUT)
                time.sleep(3)
        message = "Limit switch released. Proceeding to AUTO in 5 secs."
        self.send_status(message)
        time.sleep(5)
        self.change_mode("AUTO")
        self.send_status("Changed to AUTO mode.")

if __name__ == '__main__':
    try:
        controller = ServoController()
        controller.run_sequence()
    except rospy.ROSInterruptException:
        pass