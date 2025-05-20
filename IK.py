"""
IK.py - Inverse Kinematics for 4R Manipulator

This module provides a function to compute the joint angles [theta1, theta2, theta3, theta4]
required to place the end-effector at a desired (x, y, z) position in space.

Link lengths and constants are imported from config.py
"""

import numpy as np
from config import L1, L2, L3, L4  # Link lengths in inches or desired units

def inverse_kinematics(x: float, y: float, z: float) -> np.ndarray:
    """
    Computes inverse kinematics solution for a 4R manipulator.

    Parameters:
        x (float): Target x position of the end-effector
        y (float): Target y position of the end-effector
        z (float): Target z position of the end-effector

    Returns:
        np.ndarray: Joint angles [theta1, theta2, theta3, theta4] in degrees
    """

    offset = 180
    offset_1 = 90

    

    # Compute helper values
    rho = np.sqrt(x**2 + y**2)
    alpha = 2 * L3 * (L4 + z - L1)
    beta = L4**2 + L1**2 + z**2 + 2 * (L4 * z - L4 * L1 - z * L1)
    gamma = rho**2 + L3**2 - L2**2 + beta
    r = np.sqrt((2 * rho * L3)**2 + alpha**2)
    phi = np.arctan2(alpha, 2 * rho * L3)

    # Theta 1 (base rotation)
    t1 = np.degrees(np.arctan2(y, x)) + offset_1

    # Theta 4 (elbow)
    try:
        t4_plus = phi - np.arctan2(gamma / r, np.sqrt(1 - (gamma / r)**2))
        t4_minus = phi - np.arctan2(gamma / r, -np.sqrt(1 - (gamma / r)**2))
        t4 = np.degrees(max(abs(t4_plus), abs(t4_minus))) + offset
    except ValueError:
        # Handle invalid sqrt
        t4 = float("nan")

    # Theta 2 (shoulder)
    t4_rad = np.radians(t4 - offset)
    num = L3 * np.cos(t4_rad) + L4 + z - L1
    denom = rho - L3 * np.sin(t4_rad)
    t2 = np.degrees(np.arctan2(num, denom)) + offset

    # Theta 3 (mid link)
    t3 = np.degrees(np.pi / 2 + np.radians(t2 - offset) - np.radians(t4 - offset)) + offset

    return np.array([t1, t2, t3, t4])


if __name__ == "__main__":
    # Example test
    x, y, z = 3, 7.5, 3
    joint_angles = inverse_kinematics(x, y, z)
    print("Joint angles [deg]:", joint_angles)

