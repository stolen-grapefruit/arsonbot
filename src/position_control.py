# Basic position control for Dynamixel MX-28 motors

import time
import signal
import numpy as np
from dxl import (
    DynamixelMode,
    DynamixelModel,
    DynamixelMotorFactory,
    DynamixelIO,
)
from config import motor_1, motor_2, motor_3, motor_4


class BasicController:
    def __init__(self, motor_group):
        # Motor setup
        self.motor_group = motor_group
        self.control_freq_Hz = 30.0
        self.control_period_s = 1 / self.control_freq_Hz
        self.should_continue = True

        # Signal when script is killed
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, *_):
        # Give notice before disabling torque to avoid damaging motors
        self.should_continue = False
        print("Disabling torque in 5 seconds")
        time.sleep(5.0)
        self.motor_group.disable_torque()

    def read_current_positions(self):
        # Get current positions
        q_rad = np.asarray(list(self.motor_group.angle_rad.values()))
        q_deg = np.degrees(q_rad)
        print("Current Joint Angles [deg]:", q_deg)
        return q_deg

    def move_to_positions(self, q_desired_deg):
        # Move to desired positions
        q_desired_rad = np.deg2rad(q_desired_deg)
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.Position)
        self.motor_group.enable_torque()

        desired_pos = {
            dxl_id: q_desired_rad[i]
            for i, dxl_id in enumerate(self.motor_group.dynamixel_ids)
        }
        self.motor_group.angle_rad = desired_pos
        time.sleep(0.5)

        # Wait until joints are within 1 degree of desired position
        abs_tol = np.radians(1.0)
        while self.should_continue:
            q_rad = self.motor_group.angle_rad
            if all(abs(desired_pos[dxl_id] - q_rad[dxl_id]) < abs_tol for dxl_id in desired_pos):
                break
            time.sleep(self.control_period_s)

        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.PWM)
        self.motor_group.enable_torque()
        print("Reached target positions!")

    def run(self):
        # run method
        current_positions = self.read_current_positions()
        q_desired_deg = [180.0, 180.0, 180.0, 180.0]
        self.move_to_positions(q_desired_deg)


if __name__ == "__main__":
    # COM port and Motor IDs
    dxl_io = DynamixelIO(device_name="COM4", baud_rate=57_600)
    motor_factory = DynamixelMotorFactory(dxl_io=dxl_io, dynamixel_model=DynamixelModel.MX28)
    motor_group = motor_factory.create(3, 5)

    # Main method
    controller = BasicController(motor_group)
    controller.run()
