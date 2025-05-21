import math
import signal
import time
import numpy as np
from collections import deque
from dxl import (
    DynamixelMode,
    DynamixelModel,
    DynamixelMotorGroup,
    DynamixelMotorFactory,
    DynamixelIO,
)
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager, DCMotorModel


class PDwGravityCompensationController:
    def __init__(self, motor_group, K_P, K_D, q_initial_deg, q_desired_deg, max_duration_s=2.0):
        self.q_initial_rad = np.deg2rad(q_initial_deg)
        self.q_desired_rad = np.deg2rad(q_desired_deg)
        self.K_P = np.asarray(K_P)
        self.K_D = np.asarray(K_D)

        self.motor_group = motor_group
        self.control_freq_Hz = 30.0
        self.control_period_s = 1 / self.control_freq_Hz
        self.loop_manager = FixedFrequencyLoopManager(self.control_freq_Hz)
        self.should_continue = True
        self.max_duration_s = float(max_duration_s)

        # For saving time and position data
        self.joint_position_history = deque()
        self.time_stamps = deque()

        # Manipulator physical parameters for gravity compensation
        self.m1, self.m2 = 0.193537, 0.0156075
        self.lc1, self.lc2 = 0.0533903, 0.0281188
        self.l1 = 0.0675

        # DC Motor Model
        pwm_limits = [info.pwm_limit for info in self.motor_group.motor_info.values()]
        self.motor_model = DCMotorModel(self.control_period_s, pwm_limits=np.array(pwm_limits))

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def start_control_loop(self):
        self.go_to_home_configuration()
        start_time = time.time()

        while self.should_continue:
            q_rad = np.asarray(list(self.motor_group.angle_rad.values()))
            qdot_rad_per_s = np.asarray(list(q_rad))  # Placeholder (you can replace with actual velocity if available)

            self.joint_position_history.append(q_rad)
            self.time_stamps.append(time.time() - start_time)

            if self.time_stamps[-1] - self.time_stamps[0] > self.max_duration_s:
                self.stop()
                return

            q_error = self.q_desired_rad - q_rad
            gravity_torque = self.calc_gravity_compensation_torque(q_rad)
            u = self.K_P @ q_error - self.K_D @ qdot_rad_per_s + gravity_torque

            pwm_command = self.motor_model.calc_pwm_command(u)
            self.motor_group.pwm = {
                dxl_id: pwm_val for dxl_id, pwm_val in zip(self.motor_group.dynamixel_ids, pwm_command)
            }

            print("q [deg]:", np.degrees(q_rad))
            self.loop_manager.sleep()

        self.stop()

    def stop(self):
        self.should_continue = False
        time.sleep(2 * self.control_period_s)
        self.motor_group.disable_torque()

    def signal_handler(self, *_):
        self.stop()

    def calc_gravity_compensation_torque(self, joint_positions_rad):
        q1, q2 = joint_positions_rad
        g = 9.81
        return -np.array([
            self.m1 * g * self.lc1 * math.cos(q1) + self.m2 * g * (self.l1 * math.cos(q1) + self.lc2 * math.cos(q1 + q2)),
            self.m2 * g * self.lc2 * math.cos(q1 + q2),
        ])

    def go_to_home_configuration(self):
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.Position)
        self.motor_group.enable_torque()

        home_positions_rad = {
            dxl_id: self.q_initial_rad[i]
            for i, dxl_id in enumerate(self.motor_group.dynamixel_ids)
        }
        self.motor_group.angle_rad = home_positions_rad
        time.sleep(0.5)

        abs_tol = math.radians(1.0)
        while True:
            q_rad = self.motor_group.angle_rad
            if all(abs(home_positions_rad[dxl_id] - q_rad[dxl_id]) < abs_tol for dxl_id in home_positions_rad):
                break

        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.PWM)
        self.motor_group.enable_torque()


if __name__ == "__main__":
    # Initial and desired joint configurations (in degrees)
    q_initial = [135, 135]
    q_desired = [115, 155]

    # PD gains
    K_P = np.diag([2.43, 1.25])
    K_D = np.diag([0.00058, 0.000048])

    # Connect to motors
    dxl_io = DynamixelIO(device_name="COM4", baud_rate=57600)
    motor_factory = DynamixelMotorFactory(dxl_io=dxl_io, dynamixel_model=DynamixelModel.MX28)
    motor_group = motor_factory.create(3, 5)  # Motor IDs

    # Run the controller
    controller = PDwGravityCompensationController(
        motor_group=motor_group,
        K_P=K_P,
        K_D=K_D,
        q_initial_deg=q_initial,
        q_desired_deg=q_desired,
    )
    controller.start_control_loop()
