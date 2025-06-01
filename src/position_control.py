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
from lpb import quintic_trajectory
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
    
        # Trajectory generation
        q_current_rad = np.deg2rad(self.read_current_positions())
        q_desired_rad = np.deg2rad(q_desired_deg)
        trajectory = quintic_trajectory(0, 5, q_current_rad, q_desired_rad)


        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.Position)
        self.motor_group.enable_torque()

        desired_pos = {
            dxl_id: q_desired_rad[i]
            for i, dxl_id in enumerate(self.motor_group.dynamixel_ids)
        }
        self.motor_group.angle_rad = desired_pos
        time.sleep(0.5)

        # steps = 100
        # duration = 5.0  # seconds
        # step_time = duration / steps

        # current_pos = self.motor_group.angle_rad
        # for i in range(1, steps + 1):
        #     intermediate_pos = {
        #         dxl_id: current_pos[dxl_id] + (desired_pos[dxl_id] - current_pos[dxl_id]) * i / steps
        #         for dxl_id in desired_pos
        #     }
        #     self.motor_group.angle_rad = intermediate_pos
        #     time.sleep(step_time)

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
        # q_desired_deg = [66.18164062, 119.8828125, 33.3984375, 349.01367188] # chaos position
        # q_desired_deg = [153.10546875, 86.48437, 90.703125, 272.19726562] # upright position
        # q_desired_deg = [161.80664062, 110.390625, 13.88671875, 340.57617188]
        # q_desired_deg = [193.359375, 197.13867188, 197.578125, 153.01757812]
        q_desired_deg = [211.02539062, 104.85351562, 95.18554688, 267.45117188]
        self.move_to_positions(q_desired_deg)


if __name__ == "__main__":
    # COM port and Motor IDs
    dxl_io = DynamixelIO(device_name="COM4", baud_rate=57_600)
    motor_factory = DynamixelMotorFactory(dxl_io=dxl_io, dynamixel_model=DynamixelModel.MX28)
    motor_group = motor_factory.create(motor_1, motor_2, motor_3, motor_4)

    # Main method
    controller = BasicController(motor_group)
    controller.run()
