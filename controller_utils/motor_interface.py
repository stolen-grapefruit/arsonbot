from dxl import DynamixelIO, DynamixelMode, DynamixelModel, DynamixelMotorFactory
import numpy as np
import time
import matplotlib.pyplot as plt
from config import INITIAL_POSITION_DEG, COM_PORT


def setup_motors(control_mode="PWM", use_home_position=True):
    dxl_io = DynamixelIO(device_name=COM_PORT, baud_rate=57_600)
    motor_factory = DynamixelMotorFactory(dxl_io, DynamixelModel.MX28)
    motor_group = motor_factory.create(1, 2, 3, 4)  # Update IDs here if needed

    print("🔌 Connecting to motors...")
    motor_group.disable_torque()

    if use_home_position and INITIAL_POSITION_DEG is not None:
        print("🔄 Moving to initial joint configuration in Position mode...")
        motor_group.set_mode(DynamixelMode.Position)
        motor_group.enable_torque()

        # Send target positions
        target_rad = {
            dxl_id: np.deg2rad(deg)
            for dxl_id, deg in zip(motor_group.dynamixel_ids, INITIAL_POSITION_DEG)
        }
        print("➡️ Sending initial targets:", target_rad)
        motor_group.angle_rad = target_rad
        time.sleep(0.5)

        # Wait for convergence
        abs_tol = np.radians(1.0)
        max_wait = 2.0
        t_start = time.time()

        while time.time() - t_start < max_wait:
            current = motor_group.angle_rad
            errors = [
                abs(current[dxl_id] - target_rad[dxl_id])
                for dxl_id in motor_group.dynamixel_ids
            ]
            if all(e < abs_tol for e in errors):
                break
            time.sleep(0.05)

        print("✅ Initial position reached.")

        # Print current joint angles
        current_deg = [
            np.degrees(motor_group.angle_rad[dxl_id])
            for dxl_id in motor_group.dynamixel_ids
        ]
        print("📏 Final joint angles (deg):", np.round(current_deg, 1))

        # Switch to desired control mode
        print(f"🔀 Switching to control mode: {control_mode}")
        motor_group.disable_torque()
        if control_mode == "PWM":
            motor_group.set_mode(DynamixelMode.PWM)
        else:
            motor_group.set_mode(DynamixelMode.Position)
        motor_group.enable_torque()

        # === Try to confirm mode if API supports it ===
        try:
            for dxl_id in motor_group.dynamixel_ids:
                mode = motor_group.get_operating_mode(dxl_id)
                print(f"⚙️  Motor {dxl_id} now in mode: {mode}")
        except Exception:
            print("⚠️ Could not confirm operating mode directly (API may not support it).")

    else:
        print("⚠️ Skipping home position. Setting control mode directly.")
        motor_group.disable_torque()
        if control_mode == "PWM":
            motor_group.set_mode(DynamixelMode.PWM)
        else:
            motor_group.set_mode(DynamixelMode.Position)
        motor_group.enable_torque()

        try:
            for dxl_id in motor_group.dynamixel_ids:
                mode = motor_group.get_operating_mode(dxl_id)
                print(f"⚙️  Motor {dxl_id} now in mode: {mode}")
        except Exception:
            print("⚠️ Could not confirm operating mode directly (API may not support it).")

    return motor_group


def read_joint_states(motor_group, debug=True):
    q = np.array(list(motor_group.angle_rad.values()))
    qdot = np.array(list(motor_group.velocity_rad_per_s.values()))

    if debug:
        print("Joint Angles [rad]:", q)
        print("Joint Velocities [rad/s]:", qdot)

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
