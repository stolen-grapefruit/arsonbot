"""
image_jacobian.py - Computes an approximate Jacobian for a 4R spatial manipulator

This script estimates a task-space Jacobian used in visual servoing contexts.
It uses forward kinematics and partial derivatives, abstracted through modular FK and IK scripts.

Note: This version assumes a general 4R spatial manipulator (not planar).
"""

import numpy as np
from IK import inverse_kinematics
from FK import compute_forward_kinematics


def compute_image_jacobian(q, delta=1e-4):
    """
    Numerically estimate the image-space Jacobian for a 4R manipulator.
    Approximates how the end-effector task-space position changes with each joint.

    Parameters:
        q (np.ndarray): 1D array of joint angles [q1, q2, q3, q4] in radians
        delta (float): small perturbation for finite difference

    Returns:
        J (np.ndarray): 3x4 Jacobian (maps joint velocities to ẋ, ẏ, ż of end-effector)
    """
    n = len(q)
    J = np.zeros((3, n))

    # Current end-effector position
    pos0 = compute_forward_kinematics(q)[:3]  # assume FK returns full transform or pose

    for i in range(n):
        dq = np.zeros_like(q)
        dq[i] = delta
        pos_delta = compute_forward_kinematics(q + dq)[:3]
        J[:, i] = (pos_delta - pos0) / delta

    return J


if __name__ == "__main__":
    # Example test with placeholder joint angles
    q_test = np.radians([30, 20, -15, 10])
    J = compute_image_jacobian(q_test)
    print("Estimated task-space Jacobian (ẋ, ẏ, ż) w.r.t q:")
    print(J)
