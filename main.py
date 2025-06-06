import numpy as np
import time
import matplotlib.pyplot as plt

from config import INITIAL_POSITION_DEG, FINAL_POSITION_DEG, USE_GRAVITY_COMP
from controller_utils.motor_interface import setup_motors, read_joint_states, send_pwm_command
from controller_utils.lpb import quintic_trajectory
from controller_utils.IK import inverse_kinematics  # if needed
from controller_utils.PD_GC import PDGCController, PDControllerNoGravity
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager


# === Setup Controller ===
controller_cls = PDGCController if USE_GRAVITY_COMP else PDControllerNoGravity
controller = controller_cls()

# === Setup Motors ===
motor_group = setup_motors(control_mode="PWM")

# === Setup Trajectory ===
q0 = np.radians(INITIAL_POSITION_DEG)
qf = np.radians(FINAL_POSITION_DEG)

t0, tf = 0.0, 5.0
freq = 30  # Hz
N = int((tf - t0) * freq)
t_vec, q_traj, qd_traj, _, _ = quintic_trajectory(t0, tf, q0, qf, N)

# === Control Loop ===
print("🚀 Starting control loop...")
loop = FixedFrequencyLoopManager(freq)

joint_log = []
tau_log = []
pwm_log = []
time_log = []

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

# === Final Position Check ===
print("\n🔍 Verifying Final Joint Accuracy:")
time.sleep(3.0)  # allow settling
q_end, _ = read_joint_states(motor_group)
qf_rad = np.radians(FINAL_POSITION_DEG)

final_err_rad = np.abs(q_end - qf_rad)
print("📏 Final pos error [deg]:", np.round(np.degrees(final_err_rad), 2))
if np.all(final_err_rad < np.radians(1.0)):
    print("✅ Final position reached successfully.")
else:
    print("⚠️ Final position NOT reached.")

print("✅ Trajectory complete.")

# === Convert logs to arrays ===
joint_log = np.array(joint_log)
tau_log = np.array(tau_log)
pwm_log = np.array(pwm_log)
time_log = np.array(time_log)

# === Print Summary Stats ===
print("\n📊 Average PWM (abs):", np.round(np.mean(np.abs(pwm_log), axis=0), 1))
print("📊 Max Torque [Nm]:", np.round(np.max(np.abs(tau_log), axis=0), 3))

# === Plot Joint Positions ===
# === Plot Joint Positions with Desired Trajectory Overlay ===
fig1, axs1 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for i in range(4):
    axs1[i].plot(time_log, np.degrees(joint_log[:, i]), label=f"Joint {i+1} actual", linewidth=2)
    axs1[i].plot(time_log, np.degrees(q_traj[:, i]), '--', label=f"Joint {i+1} desired", linewidth=1)
    axs1[i].axhline(INITIAL_POSITION_DEG[i], linestyle=':', color='gray', label='Initial')
    axs1[i].axhline(FINAL_POSITION_DEG[i], linestyle=':', color='red', label='Final')
    axs1[i].set_ylabel(f"Joint {i+1} [deg]")
    axs1[i].legend()
    axs1[i].grid()
axs1[-1].set_xlabel("Time [s]")
plt.suptitle("Joint Position Tracking with Desired Trajectory Overlay")
plt.tight_layout()


# === Plot Torques and PWM ===
fig2, axs2 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for i in range(4):
    axs2[i].plot(time_log, tau_log[:, i], label=f"Torque Joint {i+1} [Nm]")
    axs2[i].plot(time_log, pwm_log[:, i], '--', label=f"PWM Joint {i+1}")
    axs2[i].set_ylabel(f"Joint {i+1}")
    axs2[i].legend()
    axs2[i].grid()
axs2[-1].set_xlabel("Time [s]")
plt.suptitle("Torque and PWM Commands Over Time")
plt.tight_layout()

if __name__ == "__main__":
    plt.show()