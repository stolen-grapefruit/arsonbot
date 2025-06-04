# motor_interface.py

from dxl import DynamixelIO, DynamixelMode, DynamixelModel, DynamixelMotorFactory
import numpy as np
from config import INITIAL_POSITION_DEG
import time


def setup_motors(control_mode="PWM", use_home_position=True):
    dxl_io = DynamixelIO(device_name="COM4", baud_rate=57_600)
    motor_factory = DynamixelMotorFactory(dxl_io, DynamixelModel.MX28)
    motor_group = motor_factory.create(1, 2, 3, 4)

    motor_group.enable_torque()

    if use_home_position and INITIAL_POSITION_DEG is not None:
        # Move to safe home position before switching to PWM mode
        print("🔄 Moving to initial joint configuration...")
        # motor_group.set_mode("Position")
        motor_group.set_mode(DynamixelMode.PWM)

        home_rad = {
            dxl_id: np.deg2rad(deg)
            for dxl_id, deg in zip(motor_group.dynamixel_ids, INITIAL_POSITION_DEG)
        }
        motor_group.angle_rad = home_rad
        time.sleep(1.0)  # Allow time to reach position

        motor_group.disable_torque()

    # motor_group.set_mode(control_mode)
    if control_mode == "PWM":
        motor_group.set_mode(DynamixelMode.PWM)

    motor_group.enable_torque()

    return motor_group


def read_joint_states(motor_group, debug=False):
    q = np.array(list(motor_group.angle_rad.values()))
    qdot = np.array(list(motor_group.velocity_rad_per_s.values()))

    if debug:
        print("📡 Joint Angles [rad]:", q)
        print("📡 Joint Velocities [rad/s]:", qdot)

    return q, qdot


def send_pwm_command(motor_group, u_pwm, limit=885):
    u_clipped = np.clip(u_pwm, -limit, limit)
    pwm_dict = {
        dxl_id: int(pwm)
        for dxl_id, pwm in zip(motor_group.dynamixel_ids, u_clipped)
    }
    motor_group.pwm = pwm_dict


def send_position_command(motor_group, q_rad):
    pos_dict = {
        dxl_id: float(pos)
        for dxl_id, pos in zip(motor_group.dynamixel_ids, q_rad)
    }
    motor_group.angle_rad = pos_dict
