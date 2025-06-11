import numpy as np
import time
import matplotlib.pyplot as plt

from config import INITIAL_POSITION_DEG, USE_GRAVITY_COMP
from controller_utils.motor_interface import setup_motors, read_joint_states, send_pwm_command
from controller_utils.lpb import quintic_trajectory
from controller_utils.PD_GC import PDGCController, PDControllerNoGravity
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager

# === Define Multiple Goal Positions ===
GOAL_POSITIONS_DEG = [
    [113.8, 142.0, 118.3, 140.2],
    [170.0, 142.0, 118.3, 140.2],
    [170.0, 180.0, 118.3, 140.2],
    [170.0, 180.0, 180.0, 90.0],
    [180, 180, 180, 90],  # Reset to upright position
]

SEGMENT_DURATION = 10.0  # seconds
TIME_SCALING = 0.17
MIN_TIME_SCALE = 0.17
CONTROL_FREQ = 30       # Hz


def build_full_trajectory(initial_deg, goals_deg, duration=SEGMENT_DURATION, freq=CONTROL_FREQ):
    q_full = []
    qd_full = []
    q_start = np.radians(initial_deg)

    for q_end_deg in goals_deg:
        joint_delta = np.rad2deg(np.max(np.abs(np.radians(q_end_deg) - q_start)))
        duration = joint_delta * TIME_SCALING
        print(f"Duration: ", duration)

        q_end = np.radians(q_end_deg)
        N = int(duration * freq)
        _, q_traj, qd_traj, _, _ = quintic_trajectory(0.0, duration, q_start, q_end, N)
        q_full.append(q_traj)
        qd_full.append(qd_traj)
        q_start = q_end  # next segment starts where previous ended

    return np.vstack(q_full), np.vstack(qd_full)


def run_full_trajectory(q_traj, qd_traj, freq=CONTROL_FREQ):
    controller_cls = PDGCController if USE_GRAVITY_COMP else PDControllerNoGravity
    controller = controller_cls()

    motor_group = setup_motors(control_mode="PWM")

    loop = FixedFrequencyLoopManager(freq)
    joint_log, tau_log, pwm_log, time_log = [], [], [], []
    start_time = time.time()

    print("Starting full continuous trajectory...")

    for i in range(len(q_traj)):
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

    # Final position check
    time.sleep(2.0)
    q_end, _ = read_joint_states(motor_group)
    q_goal = q_traj[-1]
    err = np.degrees(np.abs(q_end - q_goal))
    print("Final Position Error [deg]:", np.round(err, 2))
    return np.array(joint_log), np.array(tau_log), np.array(pwm_log), np.array(time_log), np.array(q_traj)


def plot_results(joint_log, tau_log, pwm_log, time_log, q_traj):
    fig1, axs1 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for i in range(4):
        axs1[i].plot(time_log, np.degrees(joint_log[:, i]), label=f"Joint {i+1} actual", linewidth=2)
        axs1[i].plot(time_log, np.degrees(q_traj[:, i]), '--', label=f"Joint {i+1} desired", linewidth=1)
        axs1[i].set_ylabel(f"Joint {i+1} [deg]")
        axs1[i].legend()
        axs1[i].grid()
    axs1[-1].set_xlabel("Time [s]")
    plt.suptitle("Joint Position Tracking - Full Trajectory")
    plt.tight_layout()

    fig2, axs2 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for i in range(4):
        axs2[i].plot(time_log, tau_log[:, i], label=f"Torque Joint {i+1} [Nm]")
        axs2[i].plot(time_log, pwm_log[:, i], '--', label=f"PWM Joint {i+1}")
        axs2[i].set_ylabel(f"Joint {i+1}")
        axs2[i].legend()
        axs2[i].grid()
    axs2[-1].set_xlabel("Time [s]")
    plt.suptitle("Torque and PWM Commands - Full Trajectory")
    plt.tight_layout()


if __name__ == "__main__":
    q_traj, qd_traj = build_full_trajectory(INITIAL_POSITION_DEG, GOAL_POSITIONS_DEG)
    joint_log, tau_log, pwm_log, time_log, desired_traj = run_full_trajectory(q_traj, qd_traj)
    plot_results(joint_log, tau_log, pwm_log, time_log, desired_traj)
    plt.show()
