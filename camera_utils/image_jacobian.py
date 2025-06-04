"""
image_jacobian.py - Computes an analytical Jacobian for a 4R spatial manipulator

This version uses symbolic expressions adapted from MATLAB output.
Assumes joint angles q = [q1, q2, q3, q4] and link lengths from config.py.
"""

import numpy as np
from config import a2, a3, d4


def compute_image_jacobian(q):
    """
    Computes the 3x4 linear velocity Jacobian matrix for a 4R manipulator based on analytical expressions.

    Parameters:
        q (np.ndarray): Joint angles [q1, q2, q3, q4] in radians

    Returns:
        np.ndarray: 3x4 Jacobian mapping joint velocities to end-effector (ẋ, ẏ, ż)
    """
    q1, q2, q3, q4 = q
    t1 = q1
    t2 = q2
    t3 = q3

    s1 = np.sin(t1)
    c1 = np.cos(t1)
    s23 = np.sin(t2 + t3)
    c23 = np.cos(t2 + t3)
    c2 = np.cos(t2)

    J = np.zeros((3, 4))

    J[0, 0] = d4 * s23 * s1 - s1 * (a3 * c23 + a2 * c2)
    J[0, 1] = d4 * s23 * s1 - s1 * (a3 * c23 + a2 * c2)
    J[0, 2] = -c1 * (d4 * c23 + a3 * s23)
    J[0, 3] = -d4 * c23 * c1

    J[1, 0] = c1 * (a3 * c23 + a2 * c2) - d4 * s23 * c1
    J[1, 1] = c1 * (a3 * c23 + a2 * c2) - d4 * s23 * c1
    J[1, 2] = -s1 * (d4 * c23 + a3 * s23)
    J[1, 3] = -d4 * c23 * s1

    J[2, 0] = 0
    J[2, 1] = 0
    J[2, 2] = d4 * s23 - a3 * c23
    J[2, 3] = d4 * s23

    return J


if __name__ == "__main__":
    
    q_test = np.radians([30, 20, -15, 10])
    J = compute_image_jacobian(q_test)
    print("Analytical task-space Jacobian (ẋ, ẏ, ż) w.r.t q:")
    print(J)
