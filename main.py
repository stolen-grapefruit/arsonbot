from motor_interface import setup_motors, read_joint_states, send_pwm_command
from PD_GC import PDGCController
from lpb import quintic_trajectory  # or from lbp import lpb if you use parabolic
import numpy as np
import time
from mechae263C_helpers.minilabs import FixedFrequencyLoopManager

# Setup
motor_group = setup_motors(control_mode="PWM")
controller = PDGCController()
loop = FixedFrequencyLoopManager(30)  # 30 Hz

# Load trajectory
q0 = np.radians([90, 90, 90, 90])
qf = np.radians([120, 45, 80, 60])
t0, tf = 0, 6
N = int((tf - t0) * 30)
t_vec, q_traj, qd_traj, _, _ = quintic_trajectory(t0, tf, q0, qf, N)


# Run control loop
print("🚀 Executing joint-space trajectory...")
start_time = time.time()
for i in range(N):
    q, qdot = read_joint_states(motor_group)
    q_d = q_traj[i]
    qd_d = qd_traj[i]

    tau = controller.update(q, qdot, q_d, qd_d)
    pwm = controller.torque_to_pwm(tau)
    send_pwm_command(motor_group, pwm)

    loop.sleep()

print("✅ Trajectory complete!")
