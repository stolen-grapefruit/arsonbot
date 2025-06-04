# main.py

import numpy as np
import time
import matplotlib.pyplot as plt

from config import INITIAL_POSITION_DEG, USE_GRAVITY_COMP
from motor_interface import setup_motors, read_joint_states, send_pwm_command
from lpb import quintic_trajectory
from IK import inverse_kinematics
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager

# Select controller based on gravity compensation flag
if USE_GRAVITY_COMP:
    from PD_GC import PDGCController as Controller
else:
    from PD_GC import PDControllerNoGravity as Controller

# === USER INPUT: Desired end-effector task-space position (meters) ===
# TARGET_XYZ = (0.1, 0.1, 0.1)  # Example: adjust to your candle location

# === Setup motors ===
motor_group = setup_motors(control_mode="PWM")

# === Compute joint-space goal from IK ===
q0 = np.radians(INITIAL_POSITION_DEG)
# qf_deg = inverse_kinematics(*TARGET_XYZ)
qf_deg = np.array([180, 180, 180, 180])
qf = np.radians(qf_deg)

# === Generate quintic trajectory in joint space ===
t0, tf = 0.0, 10.0
freq = 5  # Hz
N = int((tf - t0) * freq)
t_vec, q_traj, qd_traj, _, _ = quintic_trajectory(t0, tf, q0, qf, N)

# === Initialize controller ===
controller = Controller()

# === Control loop ===
print("🚀 Executing joint-space trajectory...")
loop = FixedFrequencyLoopManager(freq)
joint_log = []
time_log = []
start_time = time.time()

for i in range(N):
    q, qdot = read_joint_states(motor_group)
    q_d = q_traj[i]
    qd_d = qd_traj[i]

    tau = controller.update(q, qdot, q_d, qd_d)
    pwm = controller.torque_to_pwm(tau)
    send_pwm_command(motor_group, pwm)

    joint_log.append(q)
    time_log.append(time.time() - start_time)
    loop.sleep()

print("✅ Trajectory complete!")

# === Plot joint trajectories ===
joint_log = np.array(joint_log)
time_log = np.array(time_log)

plt.figure(figsize=(10, 6))
for i in range(joint_log.shape[1]):
    plt.plot(time_log, np.degrees(joint_log[:, i]), label=f"Joint {i+1}")
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Joint Angles Over Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
