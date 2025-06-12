import numpy as np
import time
import matplotlib.pyplot as plt

from config import INITIAL_POSITION_DEG, USE_GRAVITY_COMP, JOINT_LIMITS
from controller_utils.motor_interface import setup_motors, read_joint_states, send_pwm_command
from controller_utils.lpb import quintic_trajectory
from controller_utils.PD_GC import PDGCController, PDControllerNoGravity
from controller_utils.gravity import compute_gravity_torque
from controller_utils.IK import compute_IK
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager


# === Define Task-Space Goals via IK ===
q_goal_1, _ = compute_IK(0.243, 0.0, 0.08, -60)
q_goal_2, _ = compute_IK(0.243, 0.0, 0.2, -60)
q_goal_3, _ = compute_IK(0.243, -0.112, 0.2, -60)
q_goal_4, _ = compute_IK(0.243, -0.112, 0.09, -60)

# === Define Waypoints ===
GOAL_POSITIONS_DEG = [[180, 180, 180, 90]]
GOAL_POSITIONS_DEG.append(q_goal_1.tolist())
GOAL_POSITIONS_DEG.append(q_goal_2.tolist())
GOAL_POSITIONS_DEG.append(q_goal_3.tolist())
GOAL_POSITIONS_DEG.append(q_goal_4.tolist())
GOAL_POSITIONS_DEG.append([180, 180, 180, 90])  # Return to upright

SEGMENT_DURATION = 10.0
TIME_SCALING = 0.02
MIN_TIME_SCALE = 0.17
CONTROL_FREQ = 30


def check_joint_limits(q_traj_deg, segment_idx):
    for t, q_t in enumerate(q_traj_deg):
        for i, angle in enumerate(q_t):
            lower, upper = JOINT_LIMITS[i]
            if not (lower <= angle <= upper):
                print(f"[Violation] Segment {segment_idx}, timestep {t}, joint {i+1}: angle={angle:.2f}° is outside limits [{lower}°, {upper}°]")
                return False
    return True


def build_full_trajectory(initial_deg, goals_deg, duration=SEGMENT_DURATION, freq=CONTROL_FREQ):
    q_full = []
    qd_full = []
    q_start = np.radians(initial_deg)

    for idx, q_end_deg in enumerate(goals_deg):
        joint_delta = np.rad2deg(np.max(np.abs(np.radians(q_end_deg) - q_start)))
        duration = joint_delta * TIME_SCALING
        print(f"Duration: {duration:.2f}s")

        q_end = np.radians(q_end_deg)
        N = int(duration * freq)

        _, q_traj, qd_traj, _, _ = quintic_trajectory(0.0, duration, q_start, q_end, N)

        q_traj_deg = np.rad2deg(q_traj)
        if not check_joint_limits(q_traj_deg, idx):
            print(f"[Error] Joint limits violated in segment {idx}. Skipping this segment.")
            continue

        q_full.append(q_traj)
        qd_full.append(qd_traj)
        q_start = q_end

    return np.vstack(q_full), np.vstack(qd_full)


def run_full_trajectory(q_traj, qd_traj, freq=CONTROL_FREQ):
    controller_cls = PDGCController if USE_GRAVITY_COMP else PDControllerNoGravity
    controller = controller_cls()

    motor_group = setup_motors(control_mode="PWM")

    loop = FixedFrequencyLoopManager(freq)
    joint_log, tau_log, pwm_log, gravity_pwm_log, time_log = [], [], [], [], []
    start_time = time.time()

    print("Starting full continuous trajectory...")

    for i in range(len(q_traj)):
        q, qdot = read_joint_states(motor_group)
        q_d, qd_d = q_traj[i], qd_traj[i]
        tau = controller.update(q, qdot, q_d, qd_d)
        pwm = controller.torque_to_pwm(tau)

        # Compute gravity torque → PWM for plotting
        gravity_tau = compute_gravity_torque(q)
        gravity_pwm = controller.convert_gravity(gravity_tau)

        send_pwm_command(motor_group, pwm)

        joint_log.append(q)
        tau_log.append(tau)
        pwm_log.append(pwm)
        gravity_pwm_log.append(gravity_pwm)
        time_log.append(time.time() - start_time)

        loop.sleep()

    time.sleep(2.0)
    q_end, _ = read_joint_states(motor_group)
    q_goal = q_traj[-1]
    err = np.degrees(np.abs(q_end - q_goal))
    print("Final Position Error [deg]:", np.round(err, 2))
    time.sleep(3.0)

    return (
        np.array(joint_log),
        np.array(tau_log),
        np.array(pwm_log),
        np.array(gravity_pwm_log),
        np.array(time_log),
        np.array(q_traj)
    )


def plot_results(joint_log, tau_log, pwm_log, gravity_pwm_log, time_log, q_traj):
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
        axs2[i].plot(time_log, pwm_log[:, i], '--', label=f"PWM Total Joint {i+1}")
        axs2[i].plot(time_log, gravity_pwm_log[:, i], ':', label=f"PWM Gravity Joint {i+1}")
        axs2[i].set_ylabel(f"Joint {i+1}")
        axs2[i].legend()
        axs2[i].grid()
    axs2[-1].set_xlabel("Time [s]")
    plt.suptitle("Torque and PWM Commands - Full Trajectory")
    plt.tight_layout()


if __name__ == "__main__":
    q_traj, qd_traj = build_full_trajectory(INITIAL_POSITION_DEG, GOAL_POSITIONS_DEG)
    joint_log, tau_log, pwm_log, gravity_pwm_log, time_log, desired_traj = run_full_trajectory(q_traj, qd_traj)
    plot_results(joint_log, tau_log, pwm_log, gravity_pwm_log, time_log, desired_traj)
    plt.show()