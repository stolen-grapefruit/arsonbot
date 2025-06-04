import numpy as np
import matplotlib.pyplot as plt
# from main import qf_deg
# from config import INITIAL_POSITION_DEG

def quintic_trajectory(t0, tf, q0_vec, qf_vec, N=200):
    """
    Generates a joint-space quintic trajectory with continuous q, qd, qdd across 4 joints.

    Parameters:
        t0 (float): Start time
        tf (float): End time
        q0_vec (np.ndarray): Initial joint angles [q1_0, q2_0, q3_0, q4_0]
        qf_vec (np.ndarray): Final joint angles [q1_f, q2_f, q3_f, q4_f]
        N (int): Number of time steps

    Returns:
        t (np.ndarray): Time vector [N]
        q (np.ndarray): Joint positions [N x 4]
        qd (np.ndarray): Joint velocities [N x 4]
        qdd (np.ndarray): Joint accelerations [N x 4]
        qddd (np.ndarray): Joint jerks [N x 4]
    """

    t = np.linspace(t0, tf, N)
    tau = t - t0
    T = tf - t0

    q = np.zeros((N, 4))
    qd = np.zeros((N, 4))
    qdd = np.zeros((N, 4))
    qddd = np.zeros((N, 4))

    for j in range(4):
        q0 = q0_vec[j]
        qf = qf_vec[j]

        # Boundary conditions: start/end at rest
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = (10*(qf - q0)) / T**3
        a4 = (-15*(qf - q0)) / T**4
        a5 = (6*(qf - q0)) / T**5

        q[:, j]   = a0 + a1*tau + a2*tau**2 + a3*tau**3 + a4*tau**4 + a5*tau**5
        qd[:, j]  = a1 + 2*a2*tau + 3*a3*tau**2 + 4*a4*tau**3 + 5*a5*tau**4
        qdd[:, j] = 2*a2 + 6*a3*tau + 12*a4*tau**2 + 20*a5*tau**3
        qddd[:, j] = 6*a3 + 24*a4*tau + 60*a5*tau**2

    return t, q, qd, qdd, qddd


# === DEBUG PLOT ===
if __name__ == "__main__":
    # Example start/end joint angles [radians]
    # q0 = np.radians([5, 90, 90, 90])
    q0 = np.radians([117, 180, 96, 236])
    qf = np.radians([180, 180, 180, 180])
    
    # qf = np.radians([120, 5, 80, 60])
    t0 = 0
    tf = 8

    t, q, qd, qdd, qddd = quintic_trajectory(t0, tf, q0, qf, N=300)

    joint_names = ["Joint 1", "Joint 2", "Joint 3", "Joint 4"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for j in range(4):
        axes[j].plot(t, np.degrees(q[:, j]), label='q [deg]')
        axes[j].plot(t, np.degrees(qd[:, j]), '--', label='qd [deg/s]')
        axes[j].plot(t, np.degrees(qdd[:, j]), ':', label='qdd [deg/s²]')
        axes[j].set_ylabel(joint_names[j])
        axes[j].legend()
        axes[j].grid(True)

    axes[-1].set_xlabel("Time [s]")
    plt.suptitle("Quintic Joint Trajectories (All 4 Joints)")
    plt.tight_layout()
    plt.show()
