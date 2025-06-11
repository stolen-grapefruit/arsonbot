"""
gravity.py - Gravity compensation torques for 4R manipulator

Computes torque needed at each joint to balance gravity, using link lengths and masses from config.py.
Assumes links are in a vertical plane and the arm is affected by standard gravity.
"""

import numpy as np
from config import m1, m2, m3, m4, g, lc1, lc2, lc3, lc4, L1, L2, L3, L4
from controller_utils.FK import compute_forward_kinematics
from math import cos


def compute_gravity_torque(q):
    """
    Calculate gravity compensation torques for a 4R manipulator.

    Parameters:
        q (np.ndarray): Joint angles [q1, q2, q3, q4] in radians

    Returns:
        np.ndarray: Torque vector [tau1, tau2, tau3, tau4]
    """
    q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
    q2 = q2 - np.pi/2
    q3 = q3 - np.pi

    # link parameters
    l1, l2, l3, l4 = L1, L2, L3, L4
    m2 = 0.11
    m3 = 0.11
    # print(f"Link COMS (kg): ", lc1, lc2, lc3, lc4)

    # Torque at joint 1
    tau1 = (
        m1 * g * lc1 * cos(q1)
        + m2 * g * (l1 * cos(q1) + lc2 * cos(q1 + q2))
        + m3 * g * (l1 * cos(q1) + l2 * cos(q1 + q2) + lc3 * cos(q1 + q2 + q3))
        + m4 * g * (l1 * cos(q1) + l2 * cos(q1 + q2) + l3 * cos(q1 + q2 + q3) + lc4 * cos(q1 + q2 + q3 + q4))
    )

    # Torque at joint 2
    tau2 = (
        m2 * g * lc2 * cos(q2)
        + m3 * g * (l2 * cos(q2) + lc3 * cos(q2 + q3))
        # + m4 * g * (l2 * cos(q2) + l3 * cos(q2 + q3) + lc4 * cos(q2 + q3 + q4))
    )
    print("q2 + q3: ")
    print(q2+q3)
    print("q3: ")
    print(q3)

    # Torque at joint 3
    tau3 = (
        m3 * g * lc3 * cos(q2 + q3)
        # + m4 * g * (l3 * cos(q2 + q3) + lc4 * cos(q2 + q3 + q4))
    )

    # Torque at joint 4
    tau4 = m4 * g * lc4 * cos(q2 + q3 + q4)

    return np.array([0, tau2, tau3, 0])


if __name__ == "__main__":
    q_example = np.radians([30, 20, -15, 10])
    tau = compute_gravity_torque(q_example)
    print("Gravity torques [Nm]:", tau)

