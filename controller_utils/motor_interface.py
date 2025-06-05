from dxl import DynamixelIO, DynamixelMode, DynamixelModel, DynamixelMotorFactory
import numpy as np
import time
import matplotlib.pyplot as plt
from config import INITIAL_POSITION_DEG, COM_PORT


def setup_motors(control_mode="PWM", use_home_position=True):
    dxl_io = DynamixelIO(device_name=COM_PORT, baud_rate=57_600)
    motor_factory = DynamixelMotorFactory(dxl_io, DynamixelModel.MX28)
    motor_group = motor_factory.create(1, 2, 3, 4)
    motor_group.enable_torque()

    if use_home_position and INITIAL_POSITION_DEG is not None:
        print("🔄 Moving to initial joint configuration...")
        motor_group.set_mode(DynamixelMode.Position)

        target_rad = {
            dxl_id: np.deg2rad(deg)
            for dxl_id, deg in zip(motor_group.dynamixel_ids, INITIAL_POSITION_DEG)
        }
        print("➡️ Sending initial joint targets...")
        motor_group.angle_rad = target_rad
        time.sleep(1.0)
        print("➡️ Current joint angles (deg):",
              np.round(np.degrees([motor_group.angle_rad[dxl_id] for dxl_id in motor_group.dynamixel_ids]), 1))

        # === Log joint motion to initial position ===
        joint_log = []
        time_log = []

        abs_tol = np.radians(1.0)
        max_wait_time = 5.0
        t_start = time.time()

        while time.time() - t_start < max_wait_time:
            current_rad = motor_group.angle_rad
            joint_log.append([current_rad[dxl_id] for dxl_id in motor_group.dynamixel_ids])
            time_log.append(time.time() - t_start)

            errors = [
                abs(current_rad[dxl_id] - target_rad[dxl_id])
                for dxl_id in motor_group.dynamixel_ids
            ]
            for i, dxl_id in enumerate(motor_group.dynamixel_ids):
                print(f"Joint {i+1} error: {np.degrees(errors[i]):.2f} deg")

            if all(e < abs_tol for e in errors):
                break

            time.sleep(0.05)

        print("✅ Initial position reached.")

        # === Plot ramp-up to initial pose ===
        joint_log = np.degrees(np.array(joint_log))
        time_log = np.array(time_log)
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        for i in range(4):
            axs[i].plot(time_log, joint_log[:, i], label=f"Joint {i+1}")
            axs[i].axhline(INITIAL_POSITION_DEG[i], linestyle='--', color='red', label='Target')
            axs[i].set_ylabel("Angle [deg]")
            axs[i].legend()
            axs[i].grid()
        axs[-1].set_xlabel("Time [s]")
        plt.suptitle("Joint Position Ramp-Up to Initial Pose")
        plt.tight_layout()
        plt.show()

        # === Final convergence check ===
        final_rad = motor_group.angle_rad
        final_errors = [
            abs(final_rad[dxl_id] - target_rad[dxl_id])
            for dxl_id in motor_group.dynamixel_ids
        ]
        print("📏 Initial pos error [deg]:", np.round(np.degrees(final_errors), 2))
        if all(e < abs_tol for e in final_errors):
            print("✅ All joints within 1° of INITIAL_POSITION_DEG.")
        else:
            print("⚠️ Joints NOT fully converged to initial position.")

        motor_group.disable_torque()

    # === Switch to user-requested control mode ===
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
