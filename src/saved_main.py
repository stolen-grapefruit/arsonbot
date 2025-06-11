import numpy as np
import time
import matplotlib.pyplot as plt

from config import INITIAL_POSITION_DEG, USE_GRAVITY_COMP
from controller_utils.motor_interface import setup_motors, read_joint_states, send_pwm_command
from controller_utils.lpb import quintic_trajectory
# from controller_utils.IK import inverse_kinematics
from controller_utils.PD_GC import PDGCController, PDControllerNoGravity
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager

# === Define Multiple Goal Positions ===
GOAL_POSITIONS_DEG = [
    [160, 160, 160, 160],
    [113.8, 140.0, 118.3, 140.2],
    [170.0, 140.0, 118.3, 140.2],
    [180, 180, 180, 90], # Reset to upright position
]

# === Setup Controller ===
controller_cls = PDGCController if USE_GRAVITY_COMP else PDControllerNoGravity
controller = controller_cls()

# === Setup Motors ===
motor_group = setup_motors(control_mode="PWM")


def run_trajectory(q_start_deg, q_end_deg, duration=10.0, freq=30):
    print(f"Running trajectory: {q_start_deg} -> {q_end_deg}")
    q0 = np.radians(q_start_deg)
    qf = np.radians(q_end_deg)
    N = int(duration * freq)
    t_vec, q_traj, qd_traj, _, _ = quintic_trajectory(0.0, duration, q0, qf, N)

    loop = FixedFrequencyLoopManager(freq)
    joint_log, tau_log, pwm_log, time_log = [], [], [], []
    start_time = time.time()

    for i in range(N):
        q, qdot = read_joint_states(motor_group)
        q_d, qd_d = q_traj[i], qd_traj[i]
        tau = controller.update(q, qdot, q_d, qd_d)
        pwm = controller.torque_to_pwm(tau)
        send_pwm_command(motor_group, pwm)

        joint_log.append(q)
        tau_log.append(tau)
        pwm_log.append(pwm)
        time_log.append(time.time() - start_time)

        loop.sleep()

    # Final position verification
    time.sleep(3.0)  # allow settling
    q_end, _ = read_joint_states(motor_group)
    final_err_rad = np.abs(q_end - np.radians(q_end_deg))
    print("Final pos error [deg]:", np.round(np.degrees(final_err_rad), 2))
    if np.all(final_err_rad < np.radians(1.5)):
        print("Final position reached successfully.")
    else:
        print("Final position NOT reached.")

    return (
        np.array(joint_log),
        np.array(tau_log),
        np.array(pwm_log),
        np.array(time_log),
        np.array(np.degrees(final_err_rad))
    )


def plot_results(all_joint_logs, all_tau_logs, all_pwm_logs, all_time_logs, all_desired_trajs):
    for traj_idx, (joint_log, tau_log, pwm_log, time_log, q_traj, goal_deg) in enumerate(
        zip(all_joint_logs, all_tau_logs, all_pwm_logs, all_time_logs, all_desired_trajs, GOAL_POSITIONS_DEG)
    ):
        fig1, axs1 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        for i in range(4):
            axs1[i].plot(time_log, np.degrees(joint_log[:, i]), label=f"Joint {i+1} actual", linewidth=2)
            axs1[i].plot(time_log, np.degrees(q_traj[:, i]), '--', label=f"Joint {i+1} desired", linewidth=1)
            axs1[i].axhline(goal_deg[i], linestyle=':', color='red', label='Final')
            axs1[i].set_ylabel(f"Joint {i+1} [deg]")
            axs1[i].legend()
            axs1[i].grid()
        axs1[-1].set_xlabel("Time [s]")
        plt.suptitle(f"Trajectory {traj_idx+1}: Joint Position Tracking")
        plt.tight_layout()

        fig2, axs2 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        for i in range(4):
            axs2[i].plot(time_log, tau_log[:, i], label=f"Torque Joint {i+1} [Nm]")
            axs2[i].plot(time_log, pwm_log[:, i], '--', label=f"PWM Joint {i+1}")
            axs2[i].set_ylabel(f"Joint {i+1}")
            axs2[i].legend()
            axs2[i].grid()
        axs2[-1].set_xlabel("Time [s]")
        plt.suptitle(f"Trajectory {traj_idx+1}: Torque and PWM Commands")
        plt.tight_layout()


if __name__ == "__main__":
    current_position_deg = INITIAL_POSITION_DEG
    all_joint_logs = []
    all_tau_logs = []
    all_pwm_logs = []
    all_time_logs = []
    all_desired_trajs = []
    all_joint_errors = []

    for goal_deg in GOAL_POSITIONS_DEG:
        # input(f"\nReady to move to {goal_deg}. Press Enter to start trajectory...")

        joint_log, tau_log, pwm_log, time_log, error_deg = run_trajectory(
            current_position_deg, goal_deg
        )
        q0 = np.radians(current_position_deg)
        qf = np.radians(goal_deg)
        N = len(time_log)
        _, q_traj, _, _, _ = quintic_trajectory(0.0, N / 30.0, q0, qf, N)

        all_joint_logs.append(joint_log)
        all_tau_logs.append(tau_log)
        all_pwm_logs.append(pwm_log)
        all_time_logs.append(time_log)
        all_desired_trajs.append(q_traj)
        all_joint_errors.append(error_deg)

        q_current, qdot_current = read_joint_states(motor_group)

        current_position_deg = np.rad2deg(q_current)  # update start for next traj

    print("\nAll trajectories completed.")
    plot_results(all_joint_logs, all_tau_logs, all_pwm_logs, all_time_logs, all_desired_trajs)
    print("\nFinal Joint Errors:")
    print(all_joint_errors)
    # plt.show()
